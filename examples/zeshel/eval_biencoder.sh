python blink/biencoder/eval_biencoder.py \
  --mode train,valid,test \
  --top_k 64 \
  --eval_batch_size 128 \
  --path_to_model output/zeshel/models/pytorch_model.bin \
  --data_path data/zeshel/blink_format \
  --zeshel_path ~/KILT/original_data/zeshel \
  --output_path output/zeshel/eval
