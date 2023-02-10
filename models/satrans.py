import torch

from .submodules import *
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
from .meta_basemodel import BaseModel
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup


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

        attndrop_rate =  0.1
        drop_rate =  0.1
        use_norm =  True
        self.meta_param_size=sum([meta_dnn_hidden_units[i]*meta_dnn_hidden_units[i+1] for i in range(len(meta_dnn_hidden_units)-1)])

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.Out_linear = nn.Linear(embedding_size,embedding_size,bias=False)

        self.attn_dropout = nn.Dropout(attndrop_rate)
        self.dropout = nn.Dropout(drop_rate)

        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.mode=mode
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)
        self.Q_meta_mlp = MetaNet(hidden_dim=embedding_size,dropout=drop_rate,use_norm=use_norm,meta_dnn_hidden_units=meta_dnn_hidden_units)
        #the difference of using 'pos' or not is whether use different metanet-layernorm here
        self.K_meta_mlp = self.Q_meta_mlp if 'pos' not in self.flag else MetaNet(hidden_dim=embedding_size,dropout=drop_rate,use_norm=use_norm,meta_dnn_hidden_units=meta_dnn_hidden_units)
        self.V_meta_mlp = self.Q_meta_mlp if 'pos' not in self.flag else MetaNet(hidden_dim=embedding_size,dropout=drop_rate,use_norm=use_norm,meta_dnn_hidden_units=meta_dnn_hidden_units)
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
            if 'gate' in self.flag:
                querys = querys*mlp_params_Q.unsqueeze(1).expand(querys.shape)*2
            elif 'bilinear' in self.flag:
                pass
            else:
                querys = self.Q_meta_mlp(querys,mlp_params_Q[:,:self.meta_param_size])
        if 'K' in self.mode:
            if 'gate' in self.flag:
                keys = keys*mlp_params_K.unsqueeze(1).expand(keys.shape)*2
            elif 'bilinear' in self.flag:
                pass
            else:
                keys = self.K_meta_mlp(keys,mlp_params_K[:,:self.meta_param_size])
        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        if 'bilinear' in self.flag:
            bilinear_params = bilinear_params.reshape(-1,self.head_num,self.att_embedding_size,self.att_embedding_size).permute(1,0,2,3)
            querys = querys @ bilinear_params


        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = self.attn_dropout(F.softmax(inner_product, dim=-1))  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num
        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if 'relu' in self.flag:#
            result = self.dropout(F.relu(self.Out_linear(result)))  #
        else:
            result = self.dropout(self.Out_linear(result))  #

        if self.use_res:
            result += inputs

        result = self.layer_norm(result)#important!
        return result




class SATrans(BaseModel):
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

        super(SATrans, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
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
        self.domain_att_layer_num=domain_att_layer_num
        self.att_layer_num = att_layer_num
        self.domain_column_list = domain_column_list
        self.flag=flag


        self.domain_embedding_dim=embedding_size

        self.domain_embeddings = nn.Embedding(num_domains_list[0]+1, self.domain_embedding_dim)
        meta_dnn_hidden_units=[int(x) for x in meta_dnn_hidden_units]
        print(meta_dnn_hidden_units)#print the dimension of meta_dnn_dimensions
        meta_dnn_hidden_units=[embedding_size]+list(meta_dnn_hidden_units)
        self.meta_dnn_hidden_units=meta_dnn_hidden_units

        self.domain_int_layers = nn.ModuleList(
            [Meta_Transformer_Layer(embedding_size,meta_dnn_hidden_units,flag, meta_mode,
                                    att_head_num, att_res, device=device) for _ in range(domain_att_layer_num)])

        #the total param size of metanet, which is the output size of scenario encoder
        meta_param_size=sum([meta_dnn_hidden_units[i]*meta_dnn_hidden_units[i+1] for i in range(len(meta_dnn_hidden_units)-1)])

        if 'bilinear' in self.flag:#use bilinear instead of metanet
            print('bilinear')
            meta_param_size = (embedding_size**2)//att_head_num
        elif 'gate' in self.flag:#use gate mechanism instead of metanet
            meta_param_size = embedding_size


        map_dnn_size =  [meta_param_size]#dnn size of the scenario encoder

        if 'pos' in flag:#use position ids, i.e., layer_id, q/k/v id to generate position-sensitive scenario embedding
            print('use pos')
            self.domain_embedding_dim *= 2
            self.layerid_embeddings = nn.Embedding(domain_att_layer_num,self.domain_embedding_dim//2)
            self.qkvid_embeddings = nn.Embedding(3,self.domain_embedding_dim//2)

        if 'onlyemb' in self.flag: # only use scenario embedding without scenario encoder
            print('onlyemb')
            self.domain_embeddings = nn.Embedding(num_domains_list[0]+1, meta_param_size)
            self.domain_map_dnn_Q = lambda x:x
        else:# use scenario encoder #default this version
            self.domain_map_dnn_Q = DNN_v2(self.domain_embedding_dim, map_dnn_size)#output without relu
            self.domain_map_dnn_K = self.domain_map_dnn_Q
            self.domain_map_dnn_V = self.domain_map_dnn_Q


        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1)
        self.domain_act_fn = lambda x: F.relu(x)

        if len(self.domain_column_list)>1:# if the number of scenario features more than 1
            self.domain_feature_columns = self.get_feature_columns(linear_feature_columns,domain_column_list)
            self.domain_embedding_dict = create_embedding_matrix(self.dnn_feature_columns, init_std, sparse=False, device=device)

        self.to(device)


    def get_feature_columns(self, feature_columns, col_names):
        return [feat for feat in feature_columns if feat.name in col_names]


    def forward(self, X):
        batch_size=X.shape[0]
        #input embeddings for self-attention layers
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        #domain embedding, which is separate from above input embeddings
        domain_ids = X[:, self.feature_index[self.domain_column_list[0]][0]].long()
        domain_emb = self.domain_embeddings(domain_ids)
        if len(self.domain_column_list)>1:
            domain_embedding_list, _ = self.input_from_feature_columns(X, self.domain_feature_columns, self.domain_embedding_dict)
            domain_emb = torch.stack(domain_embedding_list, dim=-1).mean(-1).squeeze()


        logit = 0
        att_input = concat_fun(sparse_embedding_list, axis=1)

        domain_emb = self.domain_act_fn(domain_emb)



        if 'pos' not in self.flag:#all positions share the same metanet
            domain_vec_Q = self.domain_map_dnn_Q(domain_emb)
            domain_vec_K = domain_vec_V = domain_vec_Q
            domain_dnn_param_list = [domain_vec_Q,domain_vec_K,domain_vec_V]



        for layerid, layer in enumerate(self.domain_int_layers):
            if 'pos' in self.flag:#each position has position-sensitive metanet
                layerids = (torch.ones(batch_size)*layerid).long().to(X.device)
                layerid_emb = self.layerid_embeddings(layerids)
                domain_dnn_param_list=[]
                for qkvid in range(3):#QKV
                    qkvids = (torch.ones(batch_size) * qkvid).long().to(X.device)
                    qkvid_emb = self.qkvid_embeddings(qkvids)
                    all_emb = self.domain_act_fn(torch.cat([domain_emb, layerid_emb+qkvid_emb],dim=1))
                    domain_vec = self.domain_map_dnn_Q(all_emb)
                    domain_dnn_param_list.append(domain_vec)


            att_input = layer(att_input,
                            domain_dnn_param_list[0],
                            domain_dnn_param_list[1],
                            domain_dnn_param_list[2],domain_dnn_param_list[0])



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



















