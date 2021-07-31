export CUDA_VISIBLE_DEVICES=0,
export PYTHONPATH=/home/rungsiman/BLINK
python blink/biencoder/train_biencoder.py \
  --data_path data/zeshel/blink_format \
  --output_path models/zeshel/biencoder \
  --learning_rate 1e-05 \
  --num_train_epochs 5 \
  --max_context_length 128 \
  --max_cand_length 128 \
  --train_batch_size 128 \
  --eval_batch_size 64 \
  --bert_model bert-base-uncased \
  --type_optimization all_encoder_layers \
  # --data_parallel
