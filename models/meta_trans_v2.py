import torch

from .submodules import *
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
from .basemodel import BaseModel
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup


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



class Meta_Transformer_Layer(nn.Module):
    def __init__(self, embedding_size,meta_dnn_hidden_units,flag, mode='Q',head_num=2, use_res=True, scaling=True, seed=1024, device='cpu'):
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

        if 'noattndrop' in self.flag:
            self.attn_dropout = nn.Dropout(0)
        else:
            self.attn_dropout = nn.Dropout(0.1)

        if 'nodrop' in self.flag:
            dropout_rate = 0
        else:
            dropout_rate = 0.1
        self.dropout = nn.Dropout(dropout_rate)


        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.mode=mode

        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        if 'nometanorm' in flag:
            print('no meta layernorm')
            use_norm = False
        else:
            use_norm = True

        self.meta_param_size=sum([meta_dnn_hidden_units[i]*meta_dnn_hidden_units[i+1] for i in range(len(meta_dnn_hidden_units)-1)])



        self.Q_meta_mlp = MetaNet(hidden_dim=embedding_size,dropout=dropout_rate,use_norm=use_norm,meta_dnn_hidden_units=meta_dnn_hidden_units)
        if 'samemeta' in self.flag:
            self.K_meta_mlp = self.Q_meta_mlp
        else:
            self.K_meta_mlp = MetaNet(hidden_dim=embedding_size,dropout=dropout_rate,use_norm=use_norm,meta_dnn_hidden_units=meta_dnn_hidden_units)



        self.to(device)

    def forward(self, inputs,  mlp_params_Q, mlp_params_K=None, mlp_params_V=None,bilinear_params=None):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        # None F D

        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        if 'Q' in self.mode:
            querys = self.Q_meta_mlp(querys,mlp_params_Q[:,:self.meta_param_size])
        if 'K' in self.mode:
            keys = self.K_meta_mlp(keys,mlp_params_K[:,:self.meta_param_size])
        if 'V' in self.mode:
            values = self.V_meta_mlp(values,mlp_params_V[:,:self.meta_param_size])
        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        if 'bilinear' in self.flag:
            bilinear_params = mlp_params_Q[:,self.meta_param_size:].reshape(-1,self.head_num,self.att_embedding_size,self.att_embedding_size).permute(1,0,2,3)
            querys = querys @ bilinear_params

        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num
        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        result = self.dropout(self.Out_linear(result))#
        if self.use_res:
            result += inputs
        result = self.layer_norm(result)#important!
        return result



class Meta_Trans_v2(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, domain_column_list,
                 num_domains_list,  att_layer_num=2, domain_att_layer_num=1,
                 att_head_num=2,
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

        super(Meta_Trans_v2, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)

        self.use_linear = use_linear
        self.use_dnn = use_dnn
        self.num_domains_list = num_domains_list
        embedding_size = self.embedding_size
        self.use_domain_dnn_linear = use_domain_dnn_linear
        self.share_domain_dnn_across_layers=share_domain_dnn_across_layers
        field_num = len(self.embedding_dict)

        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        dense_feature_columns = [x for x in linear_feature_columns if isinstance(x, DenseFeat)]
        dnn_linear_in_feature = field_num * embedding_size + self.compute_input_dim(dense_feature_columns )
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        self.domain_column_list = domain_column_list
        self.flag=flag

        if 'dedim10' in self.flag:
            print('domain dim 10')
            self.domain_embedding_dim=10
        elif 'dedim20' in self.flag:
            print('domain dim 20')
            self.domain_embedding_dim = 20
        self.domain_embedding_dim=embedding_size


        meta_dnn_hidden_units=[int(x) for x in meta_dnn_hidden_units]
        print(meta_dnn_hidden_units)
        meta_dnn_hidden_units=[embedding_size]+list(meta_dnn_hidden_units)
        self.meta_dnn_hidden_units=meta_dnn_hidden_units
        self.domain_int_layers = nn.ModuleList(
            [Meta_Transformer_Layer(embedding_size,meta_dnn_hidden_units,flag, meta_mode,
                                    att_head_num, att_res, device=device) for _ in range(domain_att_layer_num)])


        meta_param_size=sum([meta_dnn_hidden_units[i]*meta_dnn_hidden_units[i+1] for i in range(len(meta_dnn_hidden_units)-1)])

        if 'bilinear' in self.flag:
            print('bilinear')
            meta_param_size+= (embedding_size**2)//att_head_num


        map_dnn_size =  [meta_param_size]
        domain_dnn_input_dim = embedding_size

        if 'pos' in flag:
            print('use pos')
            domain_dnn_input_dim += embedding_size
            self.layerid_embeddings = nn.Embedding(domain_att_layer_num,self.domain_embedding_dim)
            self.qkvid_embeddings = nn.Embedding(3,self.domain_embedding_dim)

        if 'onlyemb' in self.flag:
            print('onlyemb')
            self.domain_embeddings = nn.Embedding(num_domains_list[0], meta_param_size)
            self.domain_map_dnn = lambda x:x
        else:
            self.domain_embeddings = nn.Embedding(num_domains_list[0], self.domain_embedding_dim)
            self.domain_map_dnn_Q = DNN_v2(self.domain_embedding_dim, map_dnn_size)
            if 'diffqkenc' in self.flag:
                print('different QK encoder')
                self.domain_map_dnn_K = DNN_v2(self.domain_embedding_dim, map_dnn_size)
            else:
                self.domain_map_dnn_K = self.domain_map_dnn_Q


        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1)

        if 'splitenc' in self.flag:
            self.domain_map_dnn_Q_list = nn.ModuleList([DNN_v2(self.domain_embedding_dim, map_dnn_size) for _ in range(domain_att_layer_num)])
            self.domain_map_dnn_K_list = nn.ModuleList([DNN_v2(self.domain_embedding_dim, map_dnn_size) for _ in range(domain_att_layer_num)])


        self.domain_act_fn = lambda x:x if 'noencrelu' in self.flag else F.relu(x)

        if len(self.domain_column_list)>1:
            self.domain_feature_columns = self.get_feature_columns(linear_feature_columns,domain_column_list)
            self.domain_embedding_dict = create_embedding_matrix(self.dnn_feature_columns, init_std, sparse=False, device=device)


        self.to(device)

    def filter_feature_columns(self, feature_columns, filtered_col_names):
        return [feat for feat in feature_columns if feat.name not in filtered_col_names]

    def get_feature_columns(self, feature_columns, col_names):
        return [feat for feat in feature_columns if feat.name in col_names]



    def forward(self, X):
        self.X=X
        batch_size=X.shape[0]
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        domain_ids = X[:, self.feature_index[self.domain_column_list[0]][0]].long()
        domain_emb = self.domain_embeddings(domain_ids)

        if len(self.domain_column_list)>1:
            domain_embedding_list, _ = self.input_from_feature_columns(X, self.domain_feature_columns, self.domain_embedding_dict)
            #print(torch.stack(domain_embedding_list, dim=-1).shape)
            domain_emb = torch.stack(domain_embedding_list, dim=-1).mean(-1).squeeze()


        logit = 0

        att_input = concat_fun(sparse_embedding_list, axis=1)


        domain_emb = self.domain_act_fn(domain_emb)




        if 'pos' not in self.flag and 'splitenc' not in self.flag:#all positions share the same mete-mlp
            domain_vec_Q = self.domain_map_dnn_Q(domain_emb)
            if 'diffqkenc' in self.flag:
                domain_vec_K = self.domain_map_dnn_K(domain_emb)
            else:
                domain_vec_K = domain_vec_Q

            domain_dnn_param_list = [domain_vec_Q,domain_vec_K]

        #if 'test' in self.flag:
        #    print(att_input[0][0][:5])
        #    print(domain_vec_Q[0][:5])
        #    print(domain_emb[0][:5])

        for layerid, layer in enumerate(self.domain_int_layers):
            if 'pos' in self.flag:
                layerids = (torch.ones(batch_size)*layerid).long().to(X.device)
                layerid_emb = self.layerid_embeddings(layerids)
                domain_dnn_param_list=[]
                for qkvid in range(3):#QKV
                    qkvids = (torch.ones(batch_size) * qkvid).long().to(X.device)
                    qkvid_emb = self.qkvid_embeddings(qkvids)
                    all_emb = self.domain_act_fn(torch.cat([domain_emb, layerid_emb+qkvid_emb],dim=1))
                    domain_vec = self.domain_map_dnn(all_emb)
                    domain_dnn_param_list.append(domain_vec)

            if 'splitenc' in self.flag:
                domain_vec_Q = self.domain_map_dnn_Q_list[layerid](domain_emb)
                domain_vec_K = self.domain_map_dnn_K_list[layerid](domain_emb)
                domain_dnn_param_list = [domain_vec_Q, domain_vec_K, None]


            att_input = layer(att_input,
                            domain_dnn_param_list[0],
                            domain_dnn_param_list[1],
                            None)



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






