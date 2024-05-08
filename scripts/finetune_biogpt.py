import os
import argparse
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BioGptModel
from torch.utils.data import DataLoader

class prepare_dataset(object):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.num = len(self.data)
        
    def load(self):
        data_list = []
        for i in range(self.num):
            data_list.append(str(self.data['src_sen'][i]))
        return {"text": data_list}

def tokenize_data(tokenizer, dataset, max_len):
    def convert_to_features(example_batch):
        encodings = tokenizer(example_batch['text'], truncation=True, padding='max_length', 
                              max_length=max_len, return_tensors="pt")
        return {"input_ids": encodings['input_ids'], "attention_mask": encodings['attention_mask']}

    dataset = dataset.map(convert_to_features, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
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
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptModel.from_pretrained("microsoft/biogpt").to(device)
    
    file_path = f'./{args.datatype}/PLOS_{args.datatype}.csv'
    d_train = prepare_dataset(file_path)
    train_data_dic = d_train.load()
    train_dataset = Dataset.from_dict(train_data_dic)
    
    train_data = tokenize_data(tokenizer, train_dataset, max_len=1024)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

    model.save_pretrained("./bioGPT-summary/")
