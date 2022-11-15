# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import FM, DNN, concat_fun


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None,
                 flag=None,domain_column=None,num_domains=None,meta_dnn_hidden_units=(32,64,32)):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus, flag=flag, domain_column=domain_column,
                                     num_domains=num_domains, meta_dnn_hidden_units=meta_dnn_hidden_units)



        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def meta_transformation(self, X, fm_input):
        #generate domain embedding
        domain_ids = X[:, self.feature_index[self.domain_column][0]].long()
        domain_emb = self.domain_embeddings(domain_ids)
        domain_emb = F.relu(domain_emb)
        domain_vec = self.domain_map_dnn(domain_emb)

        fm_input = self.meta_net(fm_input, domain_vec)
        return fm_input


    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        logit = self.linear_model(X)
        #logit = 0
        sparse_input = torch.cat(sparse_embedding_list, dim=1)
        # *************metanet**************
        if 'metatrans' in self.flag:
            sparse_input = self.meta_transformation(X, sparse_input)
        # *************metanet**************

        if 'nofm' not in self.flag:
            if self.use_fm and len(sparse_embedding_list) > 0:
                fm_input = sparse_input
                logit += self.fm(fm_input)

        if 'nodnn' not in self.flag:
            if self.use_dnn:
                #dnn_input = combined_dnn_input(
                #    sparse_embedding_list, dense_value_list)
                if len(dense_value_list)>0:
                    dnn_dense = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
                    dnn_sparse = torch.flatten(sparse_input,start_dim=1)
                    dnn_input = concat_fun([dnn_sparse, dnn_dense])
                else:
                    dnn_input=torch.flatten(sparse_input,start_dim=1)

                dnn_output = self.dnn(dnn_input)
                dnn_logit = self.dnn_linear(dnn_output)
                logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred
