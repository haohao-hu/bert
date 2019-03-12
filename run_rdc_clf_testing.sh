export TPU_NAME=bert-tpu
pip install -U scikit-learn
export BUCKET=gs://thesis_project_hu/bert
export BERT_LARGE_DIR=$BUCKET/uncased_L-24_H-1024_A-16
python run_classifier.py   --task_name=rdc   --do_eval=true   --do_predict=true   --data_dir=./rdc_dataset   --vocab_file=gs://thesis_project_hu/bert/uncased_L-24_H-1024_A-16/vocab.txt   --bert_config_file=gs://thesis_project_hu/bert/uncased_L-24_H-1024_A-16/bert_config.json   --init_checkpoint=gs://thesis_project_hu/bert/uncased_L-24_H-1024_A-16/bert_model.ckpt   --max_seq_length=128   --train_batch_size=64   --eval_batch_size=64   --predict_batch_size=64   --learning_rate=5e-5   --num_train_epochs=5.0   --output_dir=gs://thesis_project_hu/bert/rdc_output/   --use_tpu=True   --num_tpu_cores=8   --tpu_name=bert-tpu
