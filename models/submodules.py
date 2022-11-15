import torch
import torch.nn as nn
from deepctr_torch.layers import DNN, concat_fun, InteractingLayer, activation_layer
import torch.nn.functional as F
import torch
import torch.nn as nn
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


class MetaNet(nn.Module):
    """Implements FFN equation."""
    def __init__(self, hidden_dim, dropout=0.1, use_norm=True,meta_dnn_hidden_units=(32,64,32),flag=None):
        super(MetaNet, self).__init__()
        self.use_norm=use_norm
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.meta_dnn_hidden_units=meta_dnn_hidden_units
        self.ffn_act_func = nn.ReLU()
        self.flag=flag
        if self.use_norm:
            self.ffn_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=1e-6) for _ in range(1)])


    def forward(self, x, mlp_params=None):#x BxLxd,  mlp1  Bxdx2d   mlp2  Bx2dxd
        weight_list = []
        bias_list = []
        offset = 0
        for i in range(len(self.meta_dnn_hidden_units) - 1):
            domain_weight = mlp_params[:,
                            offset:offset + self.meta_dnn_hidden_units[i] * self.meta_dnn_hidden_units[i + 1]].reshape(
                -1, self.meta_dnn_hidden_units[i], self.meta_dnn_hidden_units[i + 1])
            offset += self.meta_dnn_hidden_units[i] * self.meta_dnn_hidden_units[i + 1]
            weight_list.append(domain_weight)

        residual = x
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
            x = self.ffn_layer_norms[0](x)
        return x



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

