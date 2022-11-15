##Code for Multi-Scenario Transformer

The dataset should be processed as a pandas dataframe, and the index of domain feature should start from 1. For example, in aliccp, there are three domains, so in the domain feature, the value can only be 1/2/3.


```
--data_name: name of dataset
--model_name: name of model
--domain_col: the name of the feature indicating the domain, e.g., in aliccp dataset, feature column "301" is the domain feature
--domain_att_layer_num: Transformer layer number
--att_head_num:  the number of heads in self-attention layers
--flag:  the flag of different functions
--meta_mode:  apply meta_mlp over Q/K/V, should be a string,e.g., "QK","Q","QKV",where "QK" has the best performance 
```

- Run MetaTransformerv1 on aliccp dataset
```
python main.py --data_name alicpp --model_name Meta_Trans --seed 1028 --embedding_dim 32 --learning_rate 0.005 --domain_att_layer_num 3 --att_head_num 4 --meta_mode QK --domain_col 301 --flag metatransv1
```
- Run MetaTransformerv2 on aliccp dataset
```
python main.py --data_name alicpp --model_name Meta_Trans --seed 1028 --embedding_dim 32 --learning_rate 0.005 --domain_att_layer_num 3 --att_head_num 4 --meta_mode QK --domain_col 301 --flag metatransv2
```