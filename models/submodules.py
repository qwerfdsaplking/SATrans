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
            self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

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
            x = self.ffn_layer_norm(x)
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




test_visual_ids=[ 1453289, 42966022, 24205824, 16064524, 25503516,  3816928,
        7754202, 16947958, 41552490, 31733916,  9384867, 42806083,
        5328450, 21453215, 34663885, 17948903,  6822311,  1937201,
        4410285,   201744, 15171505,  4885268, 15800351, 13915527,
       27213000, 35978128, 28289712, 25123397, 25770247, 27379605,
       13760316, 10768337,   452273,  5274535, 35171148, 34114659,
        4660627, 23625906, 21004494, 28949905,  2653287, 21004523,
        2598530,  2058354,  4361359, 30024448, 21854660,   542100,
       16140508, 14360209,  6944232,  3417702, 24046308, 24350916,
       11629049, 11187604, 11629056, 11187612, 13184828,  7152854,
       33085373, 27794211, 24050906, 39595420,  8436213, 12633431,
       23422455, 18402585, 37030610, 27590084, 15272070, 13201925,
        6642612,  6527116,  5959166,  3252713, 42670791, 17064975,
        7826776,  6486620,  6678322,  5464538,  8589534,  4621928,
       22239583,  4621982,  1999744, 10213112, 12216345, 15436084,
       13461530, 34084101, 40170298, 34475377, 15591011, 10101370,
       15349269, 28524049, 17504711, 15349277,  3775317,  2815609,
       36539077, 42794663, 34870909, 29157348, 34870923, 29157361,
       13685414,  3031651, 33378505, 37496073, 12323623,  1473535,
       12323624,  1473550, 21577131, 18077461, 17589077,   594323,
        7054643, 27366202,  9829585,  3914425, 15037633,  3485856,
        4354720, 15268491,  6452190,   200160, 20131983, 19348389,
       27414407, 23073182,   623982, 11383393,   365066, 34383407,
       24621639, 11927453, 33979305, 24556419, 33979178, 33928232,
       34880803,  6376177,   630384, 15631931,  5951692,  7208557,
       15447211,   823224, 17987370, 20497886,  7104972, 11261277,
        4139805,  2518659, 15080545,  7805953, 12295454,  3571874,
       12295473,  3571892, 10182573,  9900270,  2777486, 16264490,
       18163604, 18088934, 39725987, 39497285,  3848595,  1177255,
        2490455,  8463106, 20776222, 22872501, 14865644, 27189783,
       27357813, 32926447, 15976511,  9037206, 16599797,  4066064,
         233709, 14820183, 39580645, 22254404, 21789170, 22254370,
       12251469, 10424663,  2228191, 13618755,  9990946,  4544418,
        8218479,  7867052, 13714297,  6830214, 13714394,  6830242,
       33087538,  8930753,  6766948, 12542448,  6766965, 12542468,
       15054031, 22726955, 42554055, 32536104,  2862756,  3599094,
        1995636,   430449, 26060671, 20681516, 16312505, 28367442,
       27610710,  1577758,  7035973,  1076148,  6505221, 12011328,
        6505225, 12011351, 35463824, 35597362, 11412430,  6738224,
       17175329, 35493826,  2708743,  6845084, 15823425, 16891493,
        7970124,  1993903,  1993917,  7970139, 14002840, 12057804,
       15992035, 23855897, 37358266, 32748952, 15288040, 13791942,
        9772821, 14922582,  3962870,  5702451, 18587646, 24862968,
       11718405, 13312948,  1424320,  3322337,  9908811, 14267302,
        4129804, 16178976,  6328636, 22669267, 22616032, 22469986,
       11918118,  4032157,  8726615, 14049473, 13442974, 15774644,
         996158, 13545838, 11434325,  5396649, 33397928, 29015019,
        7251893,  7193691,  3749412,  4288536, 19161122, 20572306,
       23708892, 11485419, 10235633,  3015786, 18315801, 10235674,
        3015864, 10235682, 18048354, 21544385, 18048367,  7588629,
       42790174, 33937873, 35901407, 33937889, 15626907, 11944843,
       21591872, 18622397,  7755823, 14457876,  4118822, 14245248]

domain_visual_ids=[2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 3, 1, 2,
       1, 2, 1, 3, 2, 1, 3, 1, 3, 1, 1, 2, 1, 2, 2, 3, 3, 1, 1, 3, 3, 1,
       1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 2, 1, 2, 1, 2, 3, 3, 1, 1, 2, 2, 1,
       2, 1, 1, 3, 2, 3, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2,
       1, 2, 2, 1, 2, 1, 2, 1, 1, 3, 3, 1, 1, 3, 2, 3, 1, 2, 1, 2, 1, 2,
       2, 1, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 1, 2, 1, 3, 2, 1,
       2, 1, 2, 3, 2, 1, 1, 3, 1, 3, 1, 3, 2, 1, 1, 2, 1, 3, 1, 3, 1, 3,
       1, 2, 1, 3, 1, 2, 2, 1, 2, 1, 2, 1, 1, 3, 3, 1, 2, 1, 1, 2, 1, 2,
       1, 2, 1, 2, 3, 2, 3, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 1, 1, 2,
       2, 1, 2, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 3, 1,
       3, 2, 3, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 2, 1, 3, 1, 1, 2, 2, 1,
       1, 2, 1, 2, 1, 2, 2, 1, 3, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 2, 1,
       1, 3, 1, 2, 2, 3, 1, 2, 1, 2, 2, 3, 2, 1, 1, 2, 3, 1, 2, 3, 1, 3,
       2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 3, 2, 3, 2, 2, 1, 2, 1, 2, 1, 1, 2,
       1, 2, 2, 1]