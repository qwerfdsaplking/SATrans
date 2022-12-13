##Code for SATrans

The processed Ali-CCP and Ali-Mama datasets can be downloaded from  **[link](https://drive.google.com/file/d/1PEQNOQqO3yTwU9WcdmeIZyfV7ViiemS9/view?usp=share_link)**. 


```
--data_name: name of dataset
--model_name: name of model
--domain_col: the name of the feature indicating the domain, e.g., in aliccp dataset, feature column "301" is the domain feature
--domain_att_layer_num: Transformer layer number
--att_head_num:  the number of heads in self-attention layers
--flag:  the flag of different functions
--meta_mode:  apply meta_mlp over Q/K/V, should be a string,e.g., "QK","Q","QKV",where "QK" has the best performance 
```

- Run SATrans(EN+MetaNet) on aliccp dataset  (Best performance)
```
python main.py --data_name alicpp --model_name Meta_Trans_Final --seed 1021 --embedding_dim 32 --learning_rate 0.005 --domain_att_layer_num 3 --att_head_num 4 --meta_mode QK --domain_col 301 --flag sota
```
- Run SATrans(ENP+MetaNet) on alimama dataset  (use --flag to set "pos") (Best performance)
```
python main.py --data_name alimama --model_name SATrans --seed 1021 --embedding_dim 32 --learning_rate 0.001 --domain_att_layer_num 3 --att_head_num 4 --meta_mode QK --domain_col 301 --flag sota-pos
```


## Environments
- NVIDIA GeForce GTX 1080 Ti
- CUDA 11.2 (For GPU)
- Python==3.7.10
- torch==1.10.0+cu113
- tensorflow==2.7.0
- deepctr-torch==0.2.9
- scikit-learn==1.0.1


