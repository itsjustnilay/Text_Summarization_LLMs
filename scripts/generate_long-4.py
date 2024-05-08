import torch
from datasets import Dataset
from transformers import LEDTokenizer, LEDForConditionalGeneration
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load tokenizer
tokenizer = LEDTokenizer.from_pretrained("./long-ft-arvix/checkpoint-250/")
model = LEDForConditionalGeneration.from_pretrained("./long-ft-arvix/checkpoint-250/").to(device)


def generate_sum(batch):
    inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to(device)
    attention_mask = inputs_dict.attention_mask.to(device)
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    
    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return batch

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
        #keyword.append(dic['keywords'])
    
    return article, lay_sum

def load_task2_data():
    data_folder = './task2_development'
    datatype = 'val'
    data_path = os.path.join(data_folder, f'{datatype}.jsonl')
    
    abstract = []
    lay_sum = []
    
    file = open(data_path, 'r')
    for line in (file.readlines()):
        dic = json.loads(line)
        abstract.append(dic['abstract'])
        lay_sum.append(dic['lay_summary'])
    
    return abstract, lay_sum

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="arxiv_pubmed")
parser.add_argument("--datatype", type=str, default="test")
args = parser.parse_args()

article_val, lay_sum_val = load_task1_data(args)
val_dataset = {'article': article_val, 'abstract': lay_sum_val}
val_dataset = Dataset.from_dict(val_dataset)
val_dataset = val_dataset.select(range(100))

result = val_dataset.map(generate_sum, batched=True, batch_size=1)
# print(result["predicted_abstract"][0])

#from write_data_csv import write_data_txt
def write_data_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

write_data_txt(result["predicted_abstract"], "long-ft-arvix-grt-AP")