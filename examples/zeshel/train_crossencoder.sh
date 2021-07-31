export CUDA_VISIBLE_DEVICES=0,
export PYTHONPATH=/home/rungsiman/BLINK
python blink/crossencoder/train_cross.py \
  --data_path  models/zeshel/top64_candidates/ \
  --output_path models/zeshel/crossencoder \
  --learning_rate 2e-05 \
  --num_train_epochs 5 \
  --max_context_length 128 \
  --max_cand_length 128 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --bert_model bert-base-uncased \
  --type_optimization all_encoder_layers \
  --add_linear \
  --zeshel True \
  --debug
  # --data_parallel \
