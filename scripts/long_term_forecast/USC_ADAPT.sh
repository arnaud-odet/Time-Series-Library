task_name=long_term_forecast
is_training=1
root_path=./dataset/USC/
data_path=na
model_id=USC_32_32
seq_len=32
pred_len=32
label_len=16
input_features=P
data=USC
features=M
d_lstm=128
lstm_layers=1
e_layers=2
d_model=128
n_heads=8
d_layers=1
d_fc=128 
factor=3
enc_in=32
dec_in=32
c_out=32
des='Exp'
batch_size=256
itr=1
dropout=0.1
learning_rate=0.001
optimizer=adamw
wd=0.05
train_epochs=20
patience=4

# ADAPT
python -u run.py \
  --model ADAPT \
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
  --input_features $input_features \
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
  --optimizer adam \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --inverse
