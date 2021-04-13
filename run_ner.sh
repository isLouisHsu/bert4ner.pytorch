#!/bin/bash

python run_ner.py \
    --version=debug \
    --device=cuda:0 \
    --n_gpu=1 \
    --task_name=ner \
    --dataset_name=weibo_ner \
    --data_dir=Chinese-NLP-Corpus/NER/Weibo/ \
    --train_file=weiboNER_2nd_conll.train \
    --dev_file=weiboNER_2nd_conll.dev \
    --test_file=weiboNER_2nd_conll.test \
    --model_type=bert_crf \
    --model_name_or_path=/media/louishsu/disk/Garage/weights/transformers/chinese-roberta-wwm-ext/ \
    --output_dir=outputs/ \
    --cache_dir=cache/ \
    --markup=bio \
    --train_max_seq_length=256 \
    --eval_max_seq_length=256 \
    --do_train \
    --do_eval \
    --do_predict \
    --do_lower_case \
    --do_adv \
    --per_gpu_train_batch_size=24 \
    --per_gpu_eval_batch_size=24 \
    --learning_rate=3e-5 \
    --other_learning_rate=1e-3 \
    --num_train_epochs=3.0 \
    --logging_steps=500 \
    --save_steps=500 \
    --overwrite_output_dir \
    --seed=42