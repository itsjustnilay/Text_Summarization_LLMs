import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import *
#from write_data_csv import write_data_txt
import tqdm

def write_data_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

# Generation candidate sentences (through beam-search)
def sen_generation(device, tokenizer, model, text: str, max_length: int, beam_nums):
    # Adjusted for T5 expected input format
    prompt = "summarize: " + text
    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=max_length, truncation=True)
    inputs = inputs.to(device)
    model = model.to(device)

    res = model.generate(
        inputs,
        length_penalty=2.0,
        num_beams=4,
        no_repeat_ngram_size=3,
        max_length=max_length,
        num_return_sequences=beam_nums,
        early_stopping=True
    )

    decode_tokens = []
    for beam_res in res:
        decode_tokens.append(tokenizer.decode(beam_res.squeeze(), skip_special_tokens=True).lower())

    return decode_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='PLOS')
    parser.add_argument("--datatype", type=str, default="val")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--beam_nums", type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    new_model = T5ForConditionalGeneration.from_pretrained("./t5-summary/")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    article, _, _ = load_task1_data(args)
    sys_out = []
    for sen in tqdm.tqdm(article):
        result = sen_generation(device, tokenizer, new_model, sen, args.max_len, args.beam_nums)
        sys_out.append(result[0])
    
    write_data_txt(sys_out, "t5_plos_summary")