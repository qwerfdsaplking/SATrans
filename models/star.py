import torch

from .submodules import *
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
from .meta_basemodel import BaseModel
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup



#The implementation of STAR
class Star_Net(BaseModel):
    """Instantiates the AutoInt Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, domain_column, num_domains, domain_id_as_feature=False, att_layer_num=3,
                 dnn_hidden_units=(256, 128),
                 use_domain_dnn = False,
                 use_domain_bn = False,
                 dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cpu', gpus=None,flag=None):

        super(Star_Net, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus,flag=flag)
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
        self.flag=flag


        self.domain_column = domain_column
        self.domain_column_list = [domain_column]
        if 'usetrans' in self.flag:
            self.int_layers = nn.ModuleList(
                [SelfAttention_Layer(embedding_size, 4, True, device=device) for _ in range(att_layer_num)])



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

            #for dnn in self.domain_dnns:
            #    self.add_regularization_weight(
            #        filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], dnn.named_parameters()), l2=l2_reg_dnn)

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
        #domain_emb = self.embedding_dict[self.domain_column](domain_ids.unsqueeze(-1))

        logit = 0

        if 'usetrans' in self.flag:
            att_input = concat_fun(sparse_embedding_list, axis=1)
            for layer in self.int_layers:
                att_input = layer(att_input)
            att_output = torch.flatten(att_input, start_dim=1)
            if len(dense_value_list)>0:
                dense_input = concat_fun(dense_value_list, axis=1)
                dnn_input = concat_fun([att_output, dense_input])
            else:
                dnn_input=att_output

        else:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)


        #deep_out = self.dnn(dnn_input)
        if self.use_domain_dnn:
            dnn_logit = torch.zeros(X.shape[0],1).to(dnn_input.device)

            for domain_id in range(self.num_domains):

                domain_dnns = self.domain_dnns[domain_id]
                domain_dnn_linear = self.domain_dnn_linears[domain_id]
                domain_dnn_input = dnn_input[domain_ids == domain_id + self.domain_id_offset]
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
                dnn_logit[domain_ids == (domain_id + self.domain_id_offset)] = domain_dnn_logit

            logit += dnn_logit
            y_pred = torch.sigmoid(logit)

        else:
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit
            y_pred = self.out(logit)

        return y_pred

