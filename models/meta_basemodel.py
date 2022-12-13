# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com
    zanshuxun, zanshuxun@aliyun.com

"""
from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from .submodules import DNN_v2,MetaNet,test_visual_ids
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
     varlen_embedding_lookup
from deepctr_torch.layers import PredictionLayer,concat_fun
from deepctr_torch.layers.utils import slice_arrays
from deepctr_torch.callbacks import History



class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu', flag=None):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    if flag and 'noembinit' in flag:
        print('do not initialize embeddings')
        pass
    else:
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None,
                 flag=None,domain_column=None,num_domains=None, meta_dnn_hidden_units=(32,64,32)):

        super(BaseModel, self).__init__()

        #******************meta module**************************
        self.flag = flag
        self.domain_column = domain_column
        self.embedding_dim = dnn_feature_columns[0].embedding_dim
        if domain_column:

            self.domain_embeddings = nn.Embedding(num_domains+1, self.embedding_dim)
            self.meta_dnn_hidden_units = meta_dnn_hidden_units
            meta_param_size=sum([meta_dnn_hidden_units[i]*meta_dnn_hidden_units[i+1] for i in range(len(meta_dnn_hidden_units)-1)])
            map_dnn_size = [meta_param_size]
            domain_dnn_input_dim = self.embedding_dim
            self.domain_map_dnn = DNN_v2(domain_dnn_input_dim, map_dnn_size)
            if 'metanorm' in self.flag:
                self.meta_net = MetaNet(hidden_dim=self.embedding_dim,use_norm=True,meta_dnn_hidden_units=meta_dnn_hidden_units)
            else:
                self.meta_net = MetaNet(hidden_dim=self.embedding_dim,use_norm=False,meta_dnn_hidden_units=meta_dnn_hidden_units)
            #******************meta module**************************





        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device, flag=flag)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

    def meta_transformation(self, X, fm_input):
        #generate domain embedding
        domain_ids = X[:, self.feature_index[self.domain_column][0]].long()
        domain_emb = self.domain_embeddings(domain_ids)
        domain_emb = F.relu(domain_emb)
        domain_vec = self.domain_map_dnn(domain_emb)

        fm_input = self.meta_net(fm_input, domain_vec)
        return fm_input

    def fit(self, x=None, y=None, batch_size=None, epochs=1, valid_cnt_per_epoch=1,verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        self.domain_id_offset = x[self.domain_column_list[0]].min()

        print('new fit')
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        steps_to_valid = steps_per_epoch//valid_cnt_per_epoch+1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            trained_num = 0
            step_num = 0
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss
                        if step_num%10==5 and 'test' in self.flag:
                            print('\ntest out: %f' %total_loss.item())
                            assert 1==0
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        trained_num+=x.shape[0]
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                            t.set_description(
                                'Training Loss %.6f,totel Loss %.6f, reg_loss %.6f, aux_loss %.6f' % (total_loss.item()/x.shape[0], total_loss_epoch/trained_num,reg_loss/x.shape[0],self.aux_loss/x.shape[0]))

                        step_num+=1
                        if valid_cnt_per_epoch>1 and step_num%steps_to_valid==0:
                            eval_result = self.evaluate(val_x, val_y, batch_size)
                            eval_str = ''
                            for name,result in eval_result.items():
                                eval_str += " - " + name + \
                                            ": {0: .4f}".format(result)
                            print(f'Step: {step_num}/{steps_per_epoch}, {eval_str}')

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size,y)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256,y=None,domain_ids=None):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        if 'showattn' in self.flag:
            self.attn_list_pos = [[0.0] * self.num_domains_list[0] for _ in range(self.domain_att_layer_num)]
            self.attn_list_neg = [[0.0] * self.num_domains_list[0] for _ in range(self.domain_att_layer_num)]
            self.attn_list_all = [[0.0] * self.num_domains_list[0] for _ in range(self.domain_att_layer_num)]
            self.inst_attn_dict = []
            offset=0

        if 'instattn' in self.flag:
            f=open(f'./inst_attn_{self.flag}.txt','w')
        with torch.no_grad():

            for _, x_test in tqdm(enumerate(test_loader)):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()



                if 'showattn' in self.flag:
                    batch_ids = [i for i in range(offset,offset+x.shape[0])]
                    for i,idx in enumerate(batch_ids):
                        break
                        if idx in test_visual_ids:
                            self.inst_attn_dict.append(model.domain_int_layers[0].normalized_att_scores.detach().cpu().numpy()[:,i,:,:])
                            print(len(self.inst_attn_dict))

                    y_label = y[offset:offset+x.shape[0]]
                    domain_ids_batch = domain_ids[offset:offset+x.shape[0]]
                    offset+=x.shape[0]
                    if domain_ids.min()==1:
                        bias=1
                    else:
                        bias=0
                    for i in range(self.domain_att_layer_num):
                        for j in range(self.num_domains_list[0]):
                            self.attn_list_pos[i][j]+=model.domain_int_layers[i].normalized_att_scores.detach()[:,(y_label==1)&(domain_ids_batch==(j+bias)),::].sum(1)
                            self.attn_list_neg[i][j]+=model.domain_int_layers[i].normalized_att_scores.detach()[:,(y_label==0)&(domain_ids_batch==(j+bias)),::].sum(1)
                            self.attn_list_all[i][j]+=model.domain_int_layers[i].normalized_att_scores.detach()[:,(domain_ids_batch==(j+bias)),::].sum(1)

                    if 'instattn' in self.flag:
                        attn=model.domain_int_layers[0].normalized_att_scores.detach()
                        for head_i in range(attn.shape[0]):
                            for inst_i in range(attn.shape[1]):
                                #print(attn.shape,attn[head_i][8][5])
                                threshold=0.2
                                if y_label[inst_i]==1 and x[inst_i][7].item()==3:
                                    if (attn[head_i][inst_i][7][5]>threshold) and (attn[head_i][inst_i][7][15]>threshold) and x[inst_i][15]>10000 and x[inst_i][7]>=2:

                                        print(7,5,15,attn[head_i][inst_i][7][5].item(),attn[head_i][inst_i][7][15].item(),y_label[inst_i].item(),y_pred[inst_i].item())
                                        print('gender',x[inst_i][5].item(),'pvalue',x[inst_i][7].item(),'shopping_level',x[inst_i][8].item(),'price',x[inst_i][15].item(),self.classes_[int(x[inst_i][15].item())])
                                        scores = attn[head_i][inst_i].detach().cpu().numpy().reshape(-1).tolist()
                                        print('----', x[inst_i].detach().cpu().numpy().tolist())

                                        s=[str(i) for i in scores]
                                        f.write(f'score {y_pred[inst_i].item()},label {y_label[inst_i].item()},pvalue {int(x[inst_i][7].item())},price {self.classes_[int(x[inst_i][15].item())]}\n')
                                        s = ','.join(s)+',\n'
                                        f.write(s)
                                        inst = x[inst_i].tolist()
                                        inst = ','.join([str(i) for i in inst])+',\n'
                                        f.write(inst)
                                        f.flush()

                                    #15,price, 5,gender,7 pvalue 8 shopping level 9 occupation
                                    if (attn[head_i][inst_i][15][7]>threshold) and ((attn[head_i][inst_i][15][5]>threshold) or (attn[head_i][inst_i][15][8]>threshold)) and x[inst_i][15]>12000 and x[inst_i][7]>=2:
                                        #print(attn[head_i])
                                        print(15,7,5,8,attn[head_i][inst_i][15][7].item(),attn[head_i][inst_i][15][5].item(),attn[head_i][inst_i][15][8].item(),y_label[inst_i].item(),y_pred[inst_i].item())
                                        print('gender',x[inst_i][5].item(),'pvalue',x[inst_i][7].item(),'shopping_level',x[inst_i][8].item(),'price',x[inst_i][15].item(),self.classes_[int(x[inst_i][15].item())])
                                        scores = attn[head_i][inst_i].detach().cpu().numpy().reshape(-1).tolist()
                                        print('----',x[inst_i].detach().cpu().numpy().tolist())

                                        #for s in scores:
                                        s=[str(i) for i in scores]
                                        f.write(f'score {y_pred[inst_i].item()},label {y_label[inst_i].item()},pvalue {int(x[inst_i][7].item())},price {self.classes_[int(x[inst_i][15].item())]}\n')
                                        s = ','.join(s)+',\n'
                                        f.write(s)
                                        inst = x[inst_i].tolist()
                                        inst = ','.join([str(i) for i in inst])+',\n'
                                        f.write(inst)
                                        f.flush()



                pred_ans.append(y_pred)


        if 'showattn' in self.flag:
            for i in range(self.domain_att_layer_num):
                for j in range(self.num_domains_list[0]):
                    self.attn_list_pos[i][j]/=((domain_ids==(j+bias))&(y==1)).sum()
                    self.attn_list_pos[i][j]=self.attn_list_pos[i][j].cpu().numpy()
                    self.attn_list_neg[i][j]/=((domain_ids==(j+bias))&(y==0)).sum()
                    self.attn_list_neg[i][j]=self.attn_list_neg[i][j].cpu().numpy()
                    self.attn_list_all[i][j]/=(domain_ids==(j+bias)).sum()
                    self.attn_list_all[i][j]=self.attn_list_all[i][j].cpu().numpy()


        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]
