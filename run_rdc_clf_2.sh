export TPU_NAME=bert-tpu
pip install -U scikit-learn
export BUCKET=gs://thesis_project_hu/bert
export BERT_LARGE_DIR=$BUCKET/uncased_L-24_H-1024_A-16
python run_classifier.py \
  --task_name=rdc \
  --do_train=true \
  --do_eval=true \
  --data_dir=./rdc_dataset \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --max_seq_length=176 \
  --train_batch_size=256 \
  --learning_rate=2.5e-4 \
  --num_train_epochs=40.0 \
  --output_dir=$BUCKET/rdc_output2/ \
  --use_tpu=True \
  --num_tpu_cores=8   \
  --tpu_name=$TPU_NAME \
  --do_predict=true
