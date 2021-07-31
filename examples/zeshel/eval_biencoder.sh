export CUDA_VISIBLE_DEVICES=1,
export PYTHONPATH=/home/rungsiman/BLINK
python blink/biencoder/eval_biencoder.py \
  --path_to_model models/zeshel/biencoder/pytorch_model.bin \
  --data_path data/zeshel/blink_format \
  --output_path models/zeshel \
  --encode_batch_size 8 \
  --eval_batch_size 1 \
  --top_k 64 \
  --save_topk_result \
  --bert_model bert-base-uncased \
  --mode train,valid,test \
  --zeshel True \
  # --data_parallel
