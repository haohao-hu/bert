pip install tqdm
#for i in {0..15}
#do
  python create_pretraining_data.py \
  --input_file=./rdc_dataset/train_sorted.txt \
  --output_file=./lm_train_corpus/val_tf_examples.tfrecord0 \
  --vocab_file=./uncased_L-24_H-1024_A-16/vocab.txt \
  --class_file=./rdc_dataset/classes.txt \
  --do_lower_case=True \
  --max_seq_length=256 \
  --max_predictions_per_seq=39 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=1 \
  --with_category
#done
