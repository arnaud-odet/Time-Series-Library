#export CUDA_VISIBLE_DEVICES=4

model_name=LSTransformer
task_name=long_term_forecast
is_training=1
root_path=./dataset/ETT-small/
data_path=ETTm2.csv
data=ETTm2
features=M
d_lstm=256
lstm_layers=1
e_layers=1
d_model=128
n_heads=8
d_layers=1
d_fc=256 
factor=3
enc_in=7
dec_in=7
c_out=7
des='Exp'
batch_size=64
itr=1
dropout=0.5

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data $data \
  --features $features \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --d_lstm $d_lstm \
  --lstm_layers $lstm_layers \
  --d_fc $d_fc \
  --dropout $dropout 


python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data $data \
  --features $features \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --d_lstm $d_lstm \
  --lstm_layers $lstm_layers \
  --d_fc $d_fc \
  --dropout $dropout 


python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data $data \
  --features $features \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --d_lstm $d_lstm \
  --lstm_layers $lstm_layers \
  --d_fc $d_fc \
  --dropout $dropout 
 

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data $data \
  --features $features \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --d_lstm $d_lstm \
  --lstm_layers $lstm_layers \
  --d_fc $d_fc \
  --dropout $dropout 
 