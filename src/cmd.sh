
python main.py --dataset music --aggregator 'sum' --n_epochs 10 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type gcn --model_type KGCN_GCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att 'uhrt_bi' --runs 3 --gpu_id 0

python main.py --dataset book --aggregator 'sum' --n_epochs 20 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 5e-5 --layer_size [64] --adj_type norm --alg_type gcn --model_type KGCN_GCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att 'uhrt_bi' --runs 3 --gpu_id 0

python main.py --dataset movie --aggregator 'sum' --n_epochs 20 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 65536 --l2_weight 5e-6 --lr 2e-2 --layer_size [32] --adj_type plain --alg_type gcmc --model_type KGCN_NGCF --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --alpha 0 --smoothing_steps 1 --pretrain 0 --att 'u_r' --runs 3 --gpu_id 0

python main.py --dataset restaurant --aggregator 'sum' --n_epochs 20 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 65536 --l2_weight 1e-7 --lr 2e-2 --layer_size [8] --adj_type norm --alg_type gcn --model_type KGCN_GCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att 'uhrt_bi' --runs 3 --gpu_id 0
