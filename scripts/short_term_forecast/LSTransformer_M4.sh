#export CUDA_VISIBLE_DEVICES=4

model_name=LSTransformer
task_name=short_term_forecast
is_training=1
root_path=./dataset/m4
data_path=ETTm1.csv
data=m4
features=M
e_layers=1
d_model=128
d_layers=1
factor=3
enc_in=1
dec_in=1
c_out=1
des='Exp'
n_heads=16
d_mode=128
batch_size=64
itr=1
d_lstm=64
lstm_layers=1
d_fc=64 
dropout=0.25
learning_rate=0.001
loss='SMAPE'

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data $data \
  --features $features \
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
  --dropout $dropout \
  --learning_rate $learning_rate \
  --loss $loss

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data $data \
  --features $features \
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
  --dropout $dropout \
  --learning_rate $learning_rate \
  --loss $loss


python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data $data \
  --features $features \
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
  --dropout $dropout \
  --learning_rate $learning_rate \
  --loss $loss
 

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data $data \
  --features $features \
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
  --dropout $dropout \
  --learning_rate $learning_rate \
  --loss $loss

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data $data \
  --features $features \
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
  --dropout $dropout \
  --learning_rate $learning_rate \
  --loss $loss

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data $data \
  --features $features \
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
  --dropout $dropout \
  --learning_rate $learning_rate \
  --loss $loss