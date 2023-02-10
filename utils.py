import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch
import numpy as np
import pickle
import random
import psutil
import h5py
import pandas as pd


def cal_ctr(data_dict):
    data_df = pd.DataFrame(data_dict)
    for col in data_df.columns:
        if data_df[col].nunique() < 20 and col != labels[0]:
            data_agg = data_df.groupby(col)['click'].agg('mean')
            # print(data_agg)
            print(col, data_agg.std())


def loadh52df(hdf5_path):
    import h5py
    print('load %s ...' % hdf5_path)
    f = h5py.File(hdf5_path, 'r')
    dic = {}
    for col in f.keys():
        dic[col] = f[col][:]
    f.close()
    return pd.DataFrame(dic)

def get_domain_feat(data, domain_col, dids_map=None):
    columns = data.keys() if isinstance(data, dict) else data.columns.tolist()
    if domain_col not in columns:
        domain_cols = domain_col.split('|')
        domain_feats = pd.DataFrame(np.concatenate([data[col].reshape(-1, 1) for col in domain_cols], axis=1),
                                    columns=domain_cols) if isinstance(data, dict) else data[domain_cols]
        domain_ids = domain_feats.apply(lambda x: tuple(x), axis=1)
        dids_set = set(domain_ids.unique().tolist())
        if not dids_map:
            dids_map = dict(zip(dids_set, range(len(dids_set))))
        domain_ids = domain_ids.map(lambda x: dids_map[x]).value
    else:
        domain_cols = [domain_col]
        domain_ids = data[domain_col]
    data[domain_col] = domain_ids
    return domain_cols, dids_map

def df2dict(df):
    dic=dict()
    for col in df.columns:
        dic[col]=df[col].values
    return dic

def list_h5(h5_path,return_f = False):
    f = h5py.File(h5_path, 'a')
    name_list=[]
    for key in f.keys():
        for k in f[key]:
            print(key+'/'+k,f[key][k].shape)
            name_list.append(key+'/'+k)
    if return_f:
        return f
    else:
        f.close()
        return name_list


def save_checkpoint(model, path):
    model.eval()

    torch.save(model.state_dict(), path)


def save_feat2hdf5(hdf5_path, data_dict):
    print('Save %s ...' % hdf5_path)
    f = h5py.File(hdf5_path, 'w')
    for k,v in data_dict.items():
        f[k] = v
    f.close()

def save_h5(file_path, key, data):
    f = h5py.File(file_path, 'a')
    if key in f.keys():
        del f[key]
    f[key] = data
    f.close()

def load_h5(file_path, key):
    f = h5py.File(file_path, 'r')
    if key in f.keys():
        return f[key][:]
        f.close()
    else:
        f.close()
        raise ValueError('there is no key in the h5 file')




def load_hdf5_feat(hdf5_path):
    print('load %s ...' % hdf5_path)
    f = h5py.File(hdf5_path, 'r')
    data_dict=dict()
    for k in f.keys():
        data_dict[k]=f[k][:]

    f.close()
    return data_dict


def get_memory_info():
    info = psutil.virtual_memory()
    print('Total memory：%.4f GB' % (info.total / 1024 / 1024 / 1024))
    print('Used memory： %.4f GB' % (info.used / 1024 / 1024 / 1024))
    print('Used percentage: %.4f' % info.percent)

def set_random_seeds(random_seed=0):
    #设置各类随机数种子
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def dump_pkl(obj,path):
    f=open(path,'wb')
    pickle.dump(obj,f)
    f.close()

def load_pkl(path):
    f=open(path,'rb')
    obj=pickle.load(f)
    f.close()
    return obj

def dump_npy(obj,path):#
    assert path[-4:]=='.npy'
    np.save(path,obj)

def load_npy(path):
    return np.load(path,allow_pickle=True)

class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs) -> None:
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                regularization: Tensor = None) -> Tensor:
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).mean()
        #regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * regularization
        else:
            regularization = 0

        return -log_prob + regularization / n_pairs

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def structured_negative_sampling4bipartite(edge_index, num_nodes: int = None,
                                 contains_neg_self_loops: bool = True):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    num_users = edge_index[0].max().item()+1

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)



    rand = torch.randint(num_users, num_nodes, (row.size(0), ), dtype=torch.long)
    neg_idx = row * num_nodes + rand

    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_users,num_nodes, (rest.size(0), ), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)


import torch.nn as nn
def get_activation_function(activation: str = 'PReLU') -> nn.Module:
    """

    :param activation:
    :return:
    """
    activation_l = activation.lower()
    if activation_l == 'relu':
        return nn.ReLU()
    elif activation_l == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif activation_l == 'prelu':
        return nn.PReLU()
    elif activation_l == 'tanh':
        return nn.Tanh()
    elif activation_l == 'selu':
        return nn.SELU()
    elif activation_l == 'elu':
        return nn.ELU()
    elif activation_l == "linear":
        return lambda x: x
    elif activation_l == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def get_aliccp_ctr_df(path, cols, k=3):
    print(cols)
    h5_path = '/home/featurize/work/data/alicpp.h5'
    f = h5py.File(h5_path, 'r')
    group = f[path]
    data_dict = {}
    for key in cols:
        if key in ['10914', '11014', '15014', '12714']:
            new_key = key + '_' + str(int(k))
        else:
            new_key = key
        data_dict[key] = group[new_key][:]
    return data_dict
