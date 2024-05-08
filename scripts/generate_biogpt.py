import argparse
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM
from tqdm import tqdm
import os
import json

def write_data_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

# Define or import your data loading function here
def load_task1_data(args):
    # Placeholder function, you need to define how data is loaded based on your requirements
    return ['Sample article text for testing.'], None, None

# Generation candidate sentences (through beam-search)
def sen_generation(device, tokenizer, model, text: str, max_length: int, beam_nums, max_new_tokens=None):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    inputs = inputs.to(device)
    model = model.to(device)

    # Determine the max_length for generation. Increase it if necessary.
    generation_length = max_length + (max_new_tokens if max_new_tokens is not None else 50)  # add 50 tokens by default

    res = model.generate(
        inputs['input_ids'],
        num_beams=beam_nums,
        no_repeat_ngram_size=3,
        max_length=generation_length,  # Updated to use a larger length for generation
        num_return_sequences=beam_nums,
        early_stopping=True,
        length_penalty=2.0
    )

    decode_tokens = []
    for beam_res in res:
        decode_tokens.append(tokenizer.decode(beam_res, skip_special_tokens=True).lower())

    return decode_tokens

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
    parser.add_argument("--datatype", type=str, default="val")
    parser.add_argument("--max_len", type=int, default=512)  # This is for tokenizing input
    parser.add_argument("--beam_nums", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=50)  # New parameter for additional generation length
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

    article, _, _ = load_task1_data(args)
    sys_out = []
    for sen in tqdm(article):
        result = sen_generation(device, tokenizer, model, sen, args.max_len, args.beam_nums, args.max_new_tokens)
        sys_out.append(result[0])

    write_data_txt(sys_out, f"{args.dataset}_{args.datatype}_output.txt")