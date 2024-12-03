model_name=LSTransformer
task_name=long_term_forecast
is_training=1
root_path=./dataset/USC/
data_path=na
input_features=V
use_action_progress=True
use_offense=True
data=USC
features=M
d_lstm=256
lstm_layers=1
e_layers=2
d_model=512
n_heads=8
d_layers=2
d_fc=256 
factor=3
enc_in=32
dec_in=32
c_out=32
des='Exp'
batch_size=64
itr=1
dropout=0.25
learning_rate=0.001
optimizer=adamw
wd=0.05
train_epochs=10
patience=3

python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id USC_24_24 \
  --model $model_name \
  --data $data \
  --input_features $input_features \
  --use_action_progress $use_action_progress \
  --use_offense $use_offense \
  --features $features \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
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
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience 