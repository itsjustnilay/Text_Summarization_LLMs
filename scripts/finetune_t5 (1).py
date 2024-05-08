import os
import json
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset

class prepare_dataset(object):
    def __init__(self, file_path, nums):
        self.data = pd.read_csv(file_path)
        # Ensure 'nums' does not exceed the number of rows in the DataFrame
        self.num = min(int(nums), len(self.data))
        
    def load(self):
        data_list = []
        for i in range(self.num):
            data_list.append("summarize: " + str(self.data['src_sen'][i]) + '\t' + str(self.data['dst_sen'][i]))
        return {"text": data_list}

def tokenize_data(tokenizer, dataset, max_len):
    def convert_to_features(example_batch):
        src_texts = []
        dst_texts = []
        for example in example_batch['text']:
            term = example.split('\t', 1)
            src_texts.append(term[0])
            dst_texts.append(term[1])
    
        src_encodings = tokenizer(src_texts, truncation=True, padding='max_length', max_length=max_len)
        dst_encodings = tokenizer(dst_texts, truncation=True, padding='max_length', max_length=max_len)
        encodings = {
            'input_ids': src_encodings['input_ids'],
            'attention_mask': src_encodings['attention_mask'],
            'labels': dst_encodings['input_ids'],
            'decoder_attention_mask': dst_encodings['attention_mask']
        }
        
        return encodings
    
    dataset = dataset.map(convert_to_features, batched=True)
    dataset = dataset.remove_columns(['text'])
    return dataset

def load_task1_data(args):
    data_folder = '.'
    data_path = os.path.join(data_folder, args.datatype)
    data_path = os.path.join(data_path, f'{args.dataset}_{args.datatype}.jsonl')

    lay_sum = []
    article = []
    keyword = []

    with open(data_path, 'r') as file:
        for line in file.readlines():
            dic = json.loads(line)
            article.append(dic['article'])
            lay_sum.append(dic['lay_summary'])
            keyword.append(dic['keywords'])
    
    return article, lay_sum, keyword

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='PLOS')
    parser.add_argument("--datatype", type=str, default="train")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    
    article, lay_summary, keywords = load_task1_data(args)
    
    d_train = prepare_dataset('./PLOS_train.csv', len(article))
    train_data_dic = d_train.load()
    train_dataset = Dataset.from_dict(train_data_dic, split='train')
    
    train_data = tokenize_data(tokenizer, train_dataset, max_len = 1024)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir='./logs',
        learning_rate=1e-4,
        logging_steps=2,
        gradient_accumulation_steps=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data
    )
    trainer.train()
    
    model.save_pretrained("./t5-summary/")
