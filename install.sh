pip install --user torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu112.html
cd work/MDR/code

pip install --user deepctr_torch

#修改端口，密码
#加载数据集
ln -s /home/featurize/data/MDR/data data