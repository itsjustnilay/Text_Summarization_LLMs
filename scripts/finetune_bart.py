#from utils import *
#!/usr/bin/env python3
#nano finetune_bart.py
import pandas as pd
import argparse
import torch
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import json

class prepare_dataset(object):
    def __init__(self, file_path, nums):
        self.data = pd.read_csv(file_path)
        self.num = int(nums)
        
    def load(self):
        data_list = []
        max_num = min(self.num, len(self.data))
        #for i in range(self.num):
        for i in range(max_num):
            data_list.append(str(self.data['src_sen'][i]) + '\t' + str(self.data['dst_sen'][i]))
        
        return {"text": data_list}

# def tokenizer            
def tokenize_data(tokenizer, dataset, max_len):
    def convert_to_features(example_batch):
        src_texts = []
        dst_texts = []
        for example in example_batch['text']:
            term = example.split('\t', 1)
            src_texts.append(term[0])
            dst_texts.append(term[1])
    
        src_encodings = tokenizer.batch_encode_plus(
            src_texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
        )
        dst_encodings = tokenizer.batch_encode_plus(
            dst_texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
        )
        encodings = {
            'input_ids': src_encodings['input_ids'],
            'attention_mask': src_encodings['attention_mask'],
            'dst_ids': dst_encodings['input_ids'],
            'target_attention_mask': dst_encodings['attention_mask']
        }
        
        return encodings
    
    dataset = dataset.map(convert_to_features, batched=True)                
    # Set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'dst_ids', 'attention_mask', 'target_attention_mask']
    dataset.with_format(type='torch', columns=columns)
    # Rename columns to the names that the forward method of the selected
    # model expects
    dataset = dataset.rename_column('dst_ids', 'labels')
    dataset = dataset.rename_column('target_attention_mask', 'decoder_attention_mask')

    # ---------------------- !!! ----------------------------------------------
    dataset = dataset.remove_columns(['text'])
    
    return dataset

def load_task1_data(args):

    data_folder = './task1_development'
    data_path = os.path.join(data_folder, args.datatype)
    data_path = os.path.join(data_path, f'{args.dataset}_{args.datatype}.jsonl')

    lay_sum = []
    article =[]

    keyword = []

    file = open(data_path, 'r')
    for line in (file.readlines()):
        dic = json.loads(line)
        article.append(dic['article'])
        lay_sum.append(dic['lay_summary'])
        keyword.append(dic['keywords'])
    
    return article, lay_sum, keyword

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='PLOS')
    parser.add_argument("--datatype", type=str, default="train")
    args = parser.parse_args()
    train_article, _, _ = load_task1_data(args)
    args.datatype = "val"
    val_article, _, _ = load_task1_data(args)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    
    d_train = prepare_dataset('PLOS_train.csv', len(train_article))
    train_data_dic = d_train.load()
    train_dataset = Dataset.from_dict(train_data_dic, split='train')
    # -------------------------------------------------------------------------
    d_valid = prepare_dataset('PLOS_val.csv', len(val_article))
    valid_data_dic = d_valid.load()
    valid_dataset = Dataset.from_dict(valid_data_dic, split='test')
    
    
    train_data = tokenize_data(tokenizer, train_dataset, max_len = 1024)
    valid_data = tokenize_data(tokenizer, valid_dataset, max_len = 512)
    
    
    # exit()
    from transformers import TrainingArguments, Trainer
        
    training_args = TrainingArguments(
    output_dir='./results',         # output directory 结果输出地址
    num_train_epochs=1,          # total # of training epochs 训练总批次
    per_device_train_batch_size=1,  # batch size per device during training 训练批大小
    per_device_eval_batch_size=1,   # batch size for evaluation 评估批大小
    logging_dir='./logs/rn_log',    # directory for storing logs 日志存储位置
    learning_rate=1e-4,             # 学习率
    save_steps=False,# 不保存检查点
    logging_steps=2,
    gradient_accumulation_steps=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.train()
    
    ##模型保存
    model.save_pretrained("./bart-3/")