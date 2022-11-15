# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
"""
import torch
import torch.nn as nn
from .basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input,build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list
from deepctr_torch.layers import DNN, concat_fun, InteractingLayer, activation_layer
from torch import Tensor
from torch.nn.modules.batchnorm import _NormBase
from torch_scatter import scatter_mean
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
import torch.nn.functional as F


class DNN_v2(nn.Module):#final layer without activation function
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN_v2, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 2)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 2)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            if i<len(self.linears)-1:
                fc = self.activation_layers[i](fc)
                fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class MDR_BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MDR_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor, shared_weight,shared_bias) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * shared_weight,
            self.bias + shared_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


class MDR_InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, num_domains, head_num=2, merge='no', use_res=True, scaling=True, seed=1024, device='cpu',flag=''):
        super(MDR_InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed
        self.merge = merge
        self.flag=flag
        self.embedding_size = embedding_size
        self.num_domains=num_domains
        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        self.domain_W_Queries = nn.ParameterList([nn.Parameter(torch.Tensor(embedding_size,embedding_size)) for i in range(num_domains)])
        self.domain_W_Keys = nn.ParameterList([nn.Parameter(torch.Tensor(embedding_size,embedding_size)) for i in range(num_domains)])

        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        self.attn_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.domain_layer_norms = nn.ModuleList([nn.LayerNorm(embedding_size, eps=1e-6) for i in range(num_domains)])

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs, domain_ids):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # None F D
        #querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        #keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        #querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        #keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        result = torch.zeros(inputs.shape).to(inputs.device)
        for i,W_Query in enumerate(self.domain_W_Queries):
            if self.merge =='qkno':
                W_Key = self.domain_W_Queries[i]
            else:
                W_Key = self.domain_W_Queries[0]
            if self.merge=='sum':
                W_Query = W_Query + self.W_Query
                W_Key = W_Key + self.W_Key

            elif self.merge=='dot':
                W_Query = W_Query * self.W_Query
                W_Key = W_Key * self.W_Key

            elif 'no' in self.merge:
                pass
            else:
                raise NotImplementedError('not implemented parameters merge')
            domain_masks = domain_ids==(i+1)
            sub_inputs = inputs[domain_masks]
            sub_querys = torch.tensordot(sub_inputs, W_Query, dims=([-1], [0]))
            sub_querys = torch.stack(torch.split(sub_querys, self.att_embedding_size, dim=2))
            sub_keys = torch.tensordot(sub_inputs, W_Key, dims=([-1], [0]))
            sub_keys = torch.stack(torch.split(sub_keys, self.att_embedding_size, dim=2))
            sub_values = values[:, domain_masks, :, :]

            inner_product = torch.einsum('bnik,bnjk->bnij', sub_querys, sub_keys)  # head_num None F F
            if self.scaling:
                inner_product /= self.att_embedding_size ** 0.5
            self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F
            sub_result = torch.matmul(self.normalized_att_scores, sub_values)  # head_num None F D/head_num

            sub_result = torch.cat(torch.split(sub_result, 1, ), dim=-1)
            sub_result = torch.squeeze(sub_result, dim=0)  # None F D
            sub_result = self.dropout(sub_result)  # =======
            if self.use_res:
                sub_result += torch.tensordot(sub_inputs, self.W_Res, dims=([-1], [0]))
            sub_result = F.relu(sub_result)
            sub_result = self.domain_layer_norms[i](sub_result)  # ===

            result[domain_masks] = sub_result


        return result


class MDR_InteractingLayer_v2(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, num_domains,num_query_bases, head_num=2, use_res=True, scaling=True, seed=1024, device='cpu'):
        super(MDR_InteractingLayer_v2, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed
        self.embedding_size = embedding_size
        self.num_query_bases = num_query_bases
        self.num_domains=num_domains

        self.base_W_Queries = nn.ParameterList([nn.Parameter(torch.Tensor(embedding_size,embedding_size)) for i in range(num_query_bases)])

        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        self.attn_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        #self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.domain_layer_norms = nn.ModuleList([nn.LayerNorm(embedding_size, eps=1e-6) for i in range(num_domains)])


        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs, domain_ids,domain_attn_scores):#batch_size * 3
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        #self.inputs=inputs
        #self.domain_ids=domain_ids
        #self.domain_attn_scores=domain_attn_scores
        querys_list = [torch.tensordot(inputs, W_Query, dims=([-1], [0])).unsqueeze(-1) for W_Query in self.base_W_Queries]
        querys = torch.cat(querys_list,dim=-1).permute(1,0,2,3)#LxBxDx3
        querys = (querys@domain_attn_scores.unsqueeze(-1)).squeeze().permute(1,0,2)
        # None F D
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F

        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        result = self.dropout(result)#=======
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        result_zeros=torch.zeros(result.shape).to(result.device)
        for i in range(self.num_domains):
            domain_masks = domain_ids == (i + 1)
            layer_norm=self.domain_layer_norms[i]
            result_zeros[domain_masks] = layer_norm(result[domain_masks])
        #result = self.layer_norm(result)#===
        return result


class SelfAttention_Layer(nn.Module):

    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=True, seed=1024, device='cpu'):
        super(SelfAttention_Layer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Out = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        self.attn_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F

        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        result = self.dropout(result)#=======
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)
        result = self.layer_norm(result)#===

        return result


class MetaPositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, hidden_dim, num_domains, dropout=0.1,use_norm=True,meta_dnn_hidden_units=(32,64,32)):

        super(MetaPositionwiseFeedForward, self).__init__()

        self.use_norm=use_norm
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.num_domains=num_domains
        self.meta_dnn_hidden_units=meta_dnn_hidden_units
        if self.use_norm:
            self.ffn_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=1e-6) for _ in range(1)])
        self.ffn_act_func = nn.ReLU()

    def forward(self, x, domain_ids=None, mlp_params=None):#x BxLxd,  mlp1  Bxdx2d   mlp2  Bx2dxd
        residual = x
        weight_list,bias_list=mlp_params
        if len(bias_list)>0:
            bias_list = [bias.unsqueeze(1).expand(bias.shape[0],x.shape[1],bias.shape[1]) for bias in bias_list]
        else:
            bias_list=[0.0]*len(weight_list)

        for i in range(len(weight_list)):

            x=x@weight_list[i]+bias_list[i]
            if i<len(weight_list)-1:
                x=self.ffn_act_func(x)
        x=self.dropout(x)
        x+=residual
        if self.use_norm:
            if domain_ids is None:
                x = self.ffn_layer_norms[0](x)
            else:
                for i in range(self.num_domains):
                    #print((domain_ids==(i+1)).sum(),i+1)
                    x[domain_ids==(i+1)]=self.ffn_layer_norms[i](x[domain_ids==(i+1)])
        return x


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, hidden_dim, ffn_hidden_dim, activation_fn="relu", dropout=0.1):

        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_act_func = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.fc2(self.ffn_act_func(self.fc1(x))))
        x += residual
        x = self.ffn_layer_norm(x)
        return x


class Transformer_Layer(nn.Module):
    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=True, seed=1024, device='cpu'):
        super(Transformer_Layer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.Out_linear = nn.Linear(embedding_size,embedding_size,bias=False)

        self.attn_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)


        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        self.ffn_layer = PositionwiseFeedForward(embedding_size, 2*embedding_size)
        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F

        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        result = self.dropout(self.Out_linear(result))#=======
        if self.use_res:
            result += inputs
        result = self.layer_norm(result)#===

        result = self.ffn_layer(result)

        return result


class MDR_Transformer_layer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, num_domains,num_query_bases, head_num=2, use_res=True, scaling=True, seed=1024, device='cpu'):
        super(MDR_Transformer_layer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed
        self.embedding_size = embedding_size
        self.num_query_bases = num_query_bases
        self.num_domains=num_domains

        self.base_W_Queries = nn.ParameterList([nn.Parameter(torch.Tensor(embedding_size,embedding_size)) for i in range(num_query_bases)])

        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.Out_linear = nn.Linear(embedding_size,embedding_size,bias=False)


        self.attn_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        #self.domain_layer_norms = nn.ModuleList([nn.LayerNorm(embedding_size, eps=1e-6) for i in range(num_domains)])

        self.ffn_layer = PositionwiseFeedForward(embedding_size, 2 * embedding_size)

        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)


        self.to(device)

    def forward(self, inputs, domain_ids,domain_attn_scores):#batch_size * 3
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        #self.inputs=inputs
        #self.domain_ids=domain_ids
        #self.domain_attn_scores=domain_attn_scores
        querys_list = [torch.tensordot(inputs, W_Query, dims=([-1], [0])).unsqueeze(-1) for W_Query in self.base_W_Queries]
        querys = torch.cat(querys_list,dim=-1).permute(1,0,2,3)#LxBxDx3
        querys = (querys@domain_attn_scores.unsqueeze(-1)).squeeze().permute(1,0,2)
        # None F D
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F

        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        result = self.dropout(self.Out_linear(result))  # =======
        if self.use_res:
            result += inputs
        result = self.layer_norm(result)  # ===
        result = self.ffn_layer(result)

        #result = self.layer_norm(result)#===
        return result

class Meta_Transformer_Layer(nn.Module):
    def __init__(self, embedding_size,num_domains,meta_dnn_hidden_units,flag, mode='Q',head_num=2, use_res=True, scaling=True, seed=1024, device='cpu'):
        super(Meta_Transformer_Layer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed
        self.flag=flag

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.Out_linear = nn.Linear(embedding_size,embedding_size,bias=False)

        self.attn_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.mode=mode
        #assert self.mode in ['Q','FFN']

        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        if 'nometanorm' in flag:
            print('no meta layernorm')
            use_norm = False
        else:
            use_norm = True

        self.Q_meta_mlp = MetaPositionwiseFeedForward(hidden_dim=embedding_size,num_domains=num_domains,use_norm=use_norm,meta_dnn_hidden_units=meta_dnn_hidden_units)

        if 'samemeta' not in self.flag:
            print('QKV have different layernorm')
            self.K_meta_mlp = MetaPositionwiseFeedForward(hidden_dim=embedding_size, num_domains=num_domains,
                                                          use_norm=use_norm,
                                                          meta_dnn_hidden_units=meta_dnn_hidden_units)
            self.V_meta_mlp = MetaPositionwiseFeedForward(hidden_dim=embedding_size, num_domains=num_domains,
                                                          use_norm=use_norm,
                                                          meta_dnn_hidden_units=meta_dnn_hidden_units)
        else:
            print('QKV have same layernorm')
            self.K_meta_mlp = self.V_meta_mlp = self.Q_meta_mlp

        self.to(device)

    def forward(self, inputs,  mlp_params_Q, mlp_params_K=None, mlp_params_V=None,domain_ids=None):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        # None F D
        if 'v2' in self.flag:
            querys = inputs
            keys = inputs
        else:#v1
            querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
            keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))

        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        if 'Q' in self.mode:
            querys = self.Q_meta_mlp(querys,domain_ids,mlp_params_Q)
        if 'K' in self.mode:
            keys = self.K_meta_mlp(keys,domain_ids,mlp_params_K)
        if 'V' in self.mode or 'novmap' in self.flag:
            values = self.V_meta_mlp(values,domain_ids,mlp_params_V)
        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F

        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num
        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        result = self.dropout(self.Out_linear(result))#=======
        if self.use_res:
            result += inputs
        result = self.layer_norm(result)#===

        return result



class Star_Trans(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, domain_column,
                 num_domains, domain_id_as_feature=False, att_layer_num=2, domain_att_layer_num=1,
                 att_head_num=2, merge='no',
                 att_layer_type='deepctr',
                 use_domain_dnn_linear = False,
                 att_res=True,
                 use_linear=True,
                 use_dnn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 flag=None,
                 task='binary', device='cpu', gpus=None):

        super(Star_Trans, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)

        #self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) >
        self.use_linear = use_linear
        self.use_dnn = use_dnn
        self.num_domains = num_domains
        self.flag=flag
        embedding_size = self.embedding_size
        self.use_domain_dnn_linear = use_domain_dnn_linear
        if domain_id_as_feature:
            field_num = len(self.embedding_dict)
        else:
            linear_feature_columns = self.filter_feature_columns(linear_feature_columns,domain_column)
            dnn_feature_columns = self.filter_feature_columns(dnn_feature_columns, domain_column)
            field_num = len(self.embedding_dict) - 1
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns

        dense_feature_columns = [x for x in linear_feature_columns if isinstance(x, DenseFeat)]

        if 'pool' in self.flag:
            dnn_linear_in_feature = embedding_size + self.compute_input_dim(dense_feature_columns)
        else:
            dnn_linear_in_feature = field_num * embedding_size + self.compute_input_dim(dense_feature_columns )


        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        self.domain_column = domain_column


        if att_layer_type=='deepctr':
            self.int_layers = nn.ModuleList(
                [SelfAttention_Layer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])
        elif att_layer_type=='trans':
            print('Transformer layers')
            self.int_layers = nn.ModuleList(
                [Transformer_Layer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])
        else:
            raise NotImplementedError(f'{att_layer_type} invalid!')

        self.domain_int_layers = nn.ModuleList(
            [MDR_InteractingLayer(embedding_size, num_domains, att_head_num, merge, att_res, flag=flag,device=device) for _ in range(domain_att_layer_num)])
        if use_domain_dnn_linear:
            self.domain_dnn_linears = nn.ModuleList([nn.Linear(dnn_linear_in_feature, 1) for _ in range(num_domains)])
        else:
            if 'outmlp' in self.flag:
                self.dnn_linear = DNN_v2(dnn_linear_in_feature, [embedding_size, 1])
            else:
                self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1)

        self.to(device)

    def filter_feature_columns(self, feature_columns, filtered_col_names):
        return [feat for feat in feature_columns if feat.name not in filtered_col_names]



    def forward(self, X):
        self.X=X
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        domain_ids = X[:, self.feature_index[self.domain_column][0]].long()

        if self.use_linear:
            logit = self.linear_model(X)# 没考虑domain as feature
        else:
            logit = 0
        #logit = 0
        att_input = concat_fun(sparse_embedding_list, axis=1)

        for layer in self.int_layers:
            att_input = layer(att_input)
        for layer in self.domain_int_layers:
            att_input = layer(att_input, domain_ids)


        if 'pool' in self.flag:
            att_output = att_input.mean(1)
        else:
            att_output = torch.flatten(att_input, start_dim=1)

        if len(dense_value_list)>0:
            dense_input = concat_fun(dense_value_list, axis=1)
            stack_out = concat_fun([att_output, dense_input])
        else:
            stack_out = att_output


        if self.use_domain_dnn_linear:
            att_logits = torch.zeros(stack_out.shape[0], 1).to(stack_out.device)
            for i in range(self.num_domains):
                domain_masks = domain_ids == (i + 1)
                att_logits[domain_masks,:] = self.domain_dnn_linears[i](stack_out[domain_masks,:])
            logit += att_logits
        else:
            logit += self.dnn_linear(stack_out)
        #deep_out = self.dnn(dnn_input)
        y_pred = torch.sigmoid(logit)

        return y_pred




class Meta_Trans(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, domain_column,
                 num_domains,  domain_id_as_feature=False, att_layer_num=2, domain_att_layer_num=1,
                 att_head_num=2,
                 att_layer_type='deepctr',
                 share_domain_dnn_across_layers=False,
                 use_domain_dnn_linear=False,
                 att_res=True,
                 use_linear=True,
                 use_dnn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu',
                 meta_dnn_hidden_units=(64,32),
                 meta_mode='Q',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cpu', gpus=None,flag=None):

        super(Meta_Trans, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus,flag=flag,domain_column=domain_column,
                                         num_domains=num_domains, meta_dnn_hidden_units=meta_dnn_hidden_units)

        self.use_linear = use_linear
        self.use_dnn = use_dnn
        embedding_size = self.embedding_size
        self.use_domain_dnn_linear = use_domain_dnn_linear
        self.share_domain_dnn_across_layers=share_domain_dnn_across_layers
        field_num = len(self.embedding_dict)

        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        dense_feature_columns = [x for x in linear_feature_columns if isinstance(x, DenseFeat)]
        self.flag=flag



        dnn_linear_in_feature = field_num * embedding_size + self.compute_input_dim(dense_feature_columns )
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        self.domain_column = domain_column


        self.domain_embeddings = nn.Embedding(num_domains+1, embedding_size)


        print(meta_dnn_hidden_units)
        meta_dnn_hidden_units=[embedding_size]+list(meta_dnn_hidden_units)
        self.meta_dnn_hidden_units=meta_dnn_hidden_units


        self.domain_int_layers = nn.ModuleList(
            [Meta_Transformer_Layer(embedding_size,num_domains,meta_dnn_hidden_units,flag, meta_mode,
                                    att_head_num, att_res, device=device) for _ in range(domain_att_layer_num)])


        meta_param_size=sum([meta_dnn_hidden_units[i]*meta_dnn_hidden_units[i+1] for i in range(len(meta_dnn_hidden_units)-1)])
        map_dnn_size =  [meta_param_size]
        domain_dnn_input_dim = embedding_size

        if 'v2' in flag:
            domain_dnn_input_dim += embedding_size
            self.layerid_embeddings = nn.Embedding(domain_att_layer_num,embedding_size)
            domain_dnn_input_dim += embedding_size
            self.qkvid_embeddings = nn.Embedding(3,embedding_size)

        self.domain_map_dnn = DNN_v2(domain_dnn_input_dim, map_dnn_size)
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1)

        self.to(device)

    def filter_feature_columns(self, feature_columns, filtered_col_names):
        return [feat for feat in feature_columns if feat.name not in filtered_col_names]


    def forward(self, X):
        self.X=X
        batch_size=X.shape[0]
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        domain_ids = X[:, self.feature_index[self.domain_column][0]].long()

        logit = 0

        att_input = concat_fun(sparse_embedding_list, axis=1)

        domain_emb = self.domain_embeddings(domain_ids)

        if 'v1' in self.flag:#all positions share the same mete-mlp
            domain_emb = F.relu(domain_emb)
            domain_vec = self.domain_map_dnn(domain_emb)
            domain_weight_list=[]
            domain_bias_list = []
            offset=0
            for i in range(len(self.meta_dnn_hidden_units)-1):
                domain_weight = domain_vec[:,offset:offset+self.meta_dnn_hidden_units[i]*self.meta_dnn_hidden_units[i+1]].reshape(-1,self.meta_dnn_hidden_units[i],self.meta_dnn_hidden_units[i+1])
                offset += self.meta_dnn_hidden_units[i]*self.meta_dnn_hidden_units[i+1]
                domain_weight_list.append(domain_weight)

        #print(att_input[0][0][:10],'~~~~~~~~~~~~~~~~~~',domain_vec[0][:10],'```',domain_emb[0][:10])


        for layerid, layer in enumerate(self.domain_int_layers):
            if 'v2' in self.flag:
                layerids = (torch.ones(batch_size)*layerid).long().to(X.device)
                layerid_emb = self.layerid_embeddings(layerids)
                domain_dnn_param_list=[]
                for qkvid in range(3):#QKV
                    qkvids = (torch.ones(batch_size) * qkvid).long().to(X.device)
                    qkvid_emb = self.qkvid_embeddings(qkvids)
                    all_emb = F.relu(torch.cat([domain_emb, layerid_emb,qkvid_emb],dim=1))
                    domain_vec = self.domain_map_dnn(all_emb)

                    domain_weight_list = []
                    domain_bias_list = []
                    offset = 0
                    for i in range(len(self.meta_dnn_hidden_units) - 1):
                        domain_weight = domain_vec[:,
                                        offset:offset + self.meta_dnn_hidden_units[i] * self.meta_dnn_hidden_units[
                                            i + 1]].reshape(-1, self.meta_dnn_hidden_units[i],
                                                            self.meta_dnn_hidden_units[i + 1])
                        offset += self.meta_dnn_hidden_units[i] * self.meta_dnn_hidden_units[i + 1]
                        domain_weight_list.append(domain_weight)

                    domain_dnn_param_list.append([domain_weight_list,domain_bias_list])

                att_input = layer(att_input, domain_dnn_param_list[0],domain_dnn_param_list[1],domain_dnn_param_list[2],domain_ids=None)

            else:#v1
                att_input = layer(att_input,
                                  [domain_weight_list,domain_bias_list],
                                  [domain_weight_list,domain_bias_list],
                                  [domain_weight_list,domain_bias_list],domain_ids=None)



        att_output = torch.flatten(att_input, start_dim=1)
        #concat sparse embeddings and dense features
        if len(dense_value_list)>0:
            dense_input = concat_fun(dense_value_list, axis=1)
            stack_out = concat_fun([att_output, dense_input])
        else:
            stack_out = att_output


        logit += self.dnn_linear(stack_out)
        y_pred = torch.sigmoid(logit)
        return y_pred




class Star_Net(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, domain_column, num_domains, domain_id_as_feature=False, att_layer_num=3,
                 dnn_hidden_units=(256, 128),
                 use_domain_dnn = False,
                 use_domain_bn = False,
                 dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cpu', gpus=None):

        super(Star_Net, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        self.num_domains = num_domains
        embedding_size = self.embedding_size
        if domain_id_as_feature:
            field_num = len(self.embedding_dict)
        else:
            linear_feature_columns = self.filter_feature_columns(linear_feature_columns,domain_column)
            dnn_feature_columns = self.filter_feature_columns(dnn_feature_columns, domain_column)
            field_num = len(self.embedding_dict) - 1
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.use_domain_dnn = use_domain_dnn
        self.use_domain_bn = use_domain_bn
        self.domain_id_as_feature = domain_id_as_feature
        self.domain_column = domain_column


        if use_domain_dnn:
            print('----=4===')
            dnn_input_dim = self.compute_input_dim(dnn_feature_columns)
            self.shared_bn_weight = nn.Parameter(torch.ones(dnn_input_dim))
            self.shared_bn_bias = nn.Parameter(torch.zeros(dnn_input_dim))
            if use_domain_bn:
                self.bns = nn.ModuleList([MDR_BatchNorm(dnn_input_dim) for i in range(num_domains)])

            self.domain_dnns = nn.ModuleList([DNN(dnn_input_dim, dnn_hidden_units,
                                                  activation=dnn_activation, l2_reg=l2_reg_dnn,
                                                  dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                                  init_std=init_std, device=device) for i in range(num_domains)])
            self.domain_dnn_linears =nn.ModuleList([nn.Linear(dnn_hidden_units[-1], 1) for i in range(num_domains)])
            #self.add_regularization_weight(
            #    filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            #self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
            self.shared_dnn = DNN(dnn_input_dim, dnn_hidden_units,
                                  activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                  use_bn=dnn_use_bn,
                                  init_std=init_std, device=device)
            self.shared_dnn_linear = nn.Linear(dnn_hidden_units[-1], 1)

        else:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.to(device)

    def filter_feature_columns(self, feature_columns, filtered_col_names):
        return [feat for feat in feature_columns if feat.name not in filtered_col_names]


    def forward(self, X):
        self.X=X
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        domain_ids = X[:, self.feature_index[self.domain_column][0]].long()
        domain_emb = self.embedding_dict[self.domain_column](domain_ids.unsqueeze(-1))

        logit = 0

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if self.use_domain_dnn:
            dnn_logit = torch.zeros(X.shape[0],1).to(dnn_input.device)

            for domain_id in range(self.num_domains):

                domain_dnns = self.domain_dnns[domain_id]
                domain_dnn_linear = self.domain_dnn_linears[domain_id]
                domain_dnn_input = dnn_input[domain_ids == domain_id + 1]
                if self.use_domain_bn:
                    domain_bn = self.bns[domain_id]
                    domain_dnn_input = domain_bn(domain_dnn_input,self.shared_bn_weight,self.shared_bn_bias)
                #domain_dnn_output = domain_dnns(domain_dnn_input)
                for i, domain_linear_i in enumerate(domain_dnns.linears):
                    shared_linear_i = self.shared_dnn.linears[i]
                    weight_i = domain_linear_i.weight * shared_linear_i.weight
                    bias_i = domain_linear_i.bias + shared_linear_i.bias
                    fc = F.linear(domain_dnn_input, weight_i, bias_i)
                    if domain_dnns.use_bn:
                        fc = domain_dnns.bn[i](fc)
                    fc = domain_dnns.activation_layers[i](fc)
                    fc = domain_dnns.dropout(fc)
                    domain_dnn_input = fc

                weight_linear = domain_dnn_linear.weight * self.shared_dnn_linear.weight
                bias_linear = domain_dnn_linear.bias + self.shared_dnn_linear.bias
                domain_dnn_logit = F.linear(domain_dnn_input, weight_linear, bias_linear)
                dnn_logit[domain_ids == domain_id + 1] = domain_dnn_logit

            logit += dnn_logit
            y_pred = torch.sigmoid(logit)

        else:
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit
            y_pred = self.out(logit)

        return y_pred






