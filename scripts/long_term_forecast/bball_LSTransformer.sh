model=LSTransformer
task_name=long_term_forecast
is_training=1
root_path=./dataset/bball/
data_path=all_data.npy
model_id=BBall
seq_len=24
pred_len=24
label_len=8
data=BBall
features=M
n_heads=16
factor=3
enc_in=22
dec_in=22
c_out=22
des='Exp'
batch_size=64
itr=1
dropout=0.1
learning_rate=0.001
optimizer=adamw
wd=0.05
train_epochs=36
patience=4


python -u run.py \
  --model $model \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --features $features \
  --data $data \
  --lstm_layers 1 \
  --d_lstm 256 \
  --e_layers 3 \
  --d_model 256 \
  --d_layers 1 \
  --d_fc 256 \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --inverse

  python -u run.py \
  --model $model \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --features $features \
  --data $data \
  --lstm_layers 1 \
  --d_lstm 256 \
  --e_layers 4 \
  --d_model 256 \
  --d_layers 1 \
  --d_fc 256 \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --inverse


python -u run.py \
  --model $model \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --features $features \
  --data $data \
  --lstm_layers 2 \
  --d_lstm 256 \
  --e_layers 3 \
  --d_model 256 \
  --d_layers 1 \
  --d_fc 256 \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --inverse

  python -u run.py \
  --model $model \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --features $features \
  --data $data \
  --lstm_layers 2 \
  --d_lstm 256 \
  --e_layers 4 \
  --d_model 256 \
  --d_layers 1 \
  --d_fc 256 \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --inverse

python -u run.py \
  --model $model \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --features $features \
  --data $data \
  --lstm_layers 3 \
  --d_lstm 256 \
  --e_layers 3 \
  --d_model 256 \
  --d_layers 1 \
  --d_fc 256 \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --inverse

  python -u run.py \
  --model $model \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --features $features \
  --data $data \
  --lstm_layers 3 \
  --d_lstm 256 \
  --e_layers 4 \
  --d_model 256 \
  --d_layers 1 \
  --d_fc 256 \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --inverse