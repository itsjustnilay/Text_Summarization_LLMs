import argparse
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from utils import *
#from write_data_csv import write_data_txt

# Write the new function here if you are not creating a separate file
def write_data_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

# generation candidate sentences (through beam-search)
def sen_generation(device, tokenizer, model, text: str, max_length: int, beam_nums):
    inputs = tokenizer.encode(text, padding=True, max_length=max_length, truncation=True,
                                return_tensors='pt')
    inputs = inputs.to(device)
    model = model.to(device)

    res = model.generate(
        inputs, length_penalty = 2, num_beams = 4, no_repeat_ngram_size = 3, 
        max_length = max_length, num_return_sequences = beam_nums 
    )
    
    decode_tokens = []
    for beam_res in res:
        decode_tokens.append(tokenizer.decode(beam_res.squeeze(), skip_special_tokens = True).lower())
        
    return decode_tokens

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
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default='PLOS')
        parser.add_argument("--datatype", type=str, default="val")
        parser.add_argument("--max_len", type=int, default=512)
        parser.add_argument("--beam_nums", type=int, default=1)
        args = parser.parse_args()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        new_model = BartForConditionalGeneration.from_pretrained("./bart-3/")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        
        article, _, _ = load_task1_data(args)
        sys_out = []
        for sen in tqdm(article):
            # generate candidate sentences list
            result = sen_generation(device, tokenizer, new_model, sen,
                                    args.max_len, args.beam_nums)
            sys_out.append(result[0])
        
        write_data_txt(sys_out, "bart_plos_1")