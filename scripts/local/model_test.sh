model_name=Pyraformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/USC/ \
  --data_path na \
  --model_id USC_32_32 \
  --model $model_name \
  --data USC \
  --features S \
  --seq_len 32 \
  --label_len 16 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 61 \
  --dec_in 61 \
  --c_out 61 \
  --des 'Exp' \
  --batch_size 128 \
  --itr 1
