task_name=long_term_forecast
is_training=1
root_path=./dataset/USC/
data_path=na
model_id=USC_32_32
seq_len=32
pred_len=32
label_len=16
features=MS
input_features=A
enc_in=61
dec_in=1
c_out=1
data=USC
e_layers=1
d_model=256
n_heads=16
d_layers=1
d_ff=512 
factor=3
des='Exp'
batch_size=32
itr=1
dropout=0.25
learning_rate=0.001
optimizer=adamw
wd=0.05
train_epochs=36
patience=12

# Transformer
python -u run.py \
  --model SABFormer \
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
  --d_ff $d_ff \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --optimizer $optimizer \
  --wd $wd \
  --train_epochs $train_epochs \
  --patience $patience \
  --embed fixed \
  --consider_only_offense \
  --inverse
