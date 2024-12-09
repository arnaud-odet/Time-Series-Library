task_name=long_term_forecast
is_training=1
root_path=./dataset/USC/
data_path=na
input_features=P
use_action_progress=True
use_offense=True
data=USC
features=M
d_lstm=64
lstm_layers=1
e_layers=1
d_model=64
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
dropout=0.1
learning_rate=0.001
optimizer=adam
wd=0.05
train_epochs=10
patience=3

#NonStationary Transformer
python -u run.py \ 
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id USC_64_64 \
  --model Nonstationary_Transformer \
  --data $data \
  --input_features $input_features \
  --use_action_progress $use_action_progress \
  --use_offense $use_offense \
  --features $features \
  --seq_len 64 \
  --label_len 12 \
  --pred_len 64 \
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
  --embed fixed

# TimesNet
python -u run.py \ 
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id USC_64_64 \
  --model TimesNet \
  --data $data \
  --input_features $input_features \
  --use_action_progress $use_action_progress \
  --use_offense $use_offense \
  --features $features \
  --seq_len 64 \
  --label_len 12 \
  --pred_len 64 \
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
  --patience $patience \
  --embed fixed

# PatchTST
python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id USC_64_64 \
  --model PatchTST \
  --data $data \
  --input_features $input_features \
  --use_action_progress $use_action_progress \
  --use_offense $use_offense \
  --features $features \
  --seq_len 64 \
  --label_len 12 \
  --pred_len 64 \
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
  --embed fixed

# LSTransformer
python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id USC_64_64 \
  --model LSTransformer \
  --data $data \
  --input_features $input_features \
  --use_action_progress $use_action_progress \
  --use_offense $use_offense \
  --features $features \
  --seq_len 64 \
  --label_len 12 \
  --pred_len 64 \
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
  --embed fixed

# iTransformer
python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id USC_64_64 \
  --model iTransformer \
  --data $data \
  --input_features $input_features \
  --use_action_progress $use_action_progress \
  --use_offense $use_offense \
  --features $features \
  --seq_len 64 \
  --label_len 12 \
  --pred_len 64 \
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
  --embed fixed

# TimeMixer
python -u run.py \
  --task_name $task_name \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id USC_64_64 \
  --model TimeMixer \
  --data $data \
  --input_features $input_features \
  --use_action_progress $use_action_progress \
  --use_offense $use_offense \
  --features $features \
  --seq_len 64 \
  --label_len 12 \
  --pred_len 64 \
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
  --embed fixed
