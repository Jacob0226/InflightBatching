import argparse
from transformers import AutoTokenizer
import os
import math

def truncate_text_to_token_count(article, target_token_count):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B", legacy=True)

    # Encode the full article to count tokens
    tokens = tokenizer.encode(article)

    # Truncate to the target token count
    if len(tokens) > target_token_count:
        tokens = tokens[:target_token_count]

    # Decode the tokens back to text
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return truncated_text

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Truncate text to a specific token count.")
    parser.add_argument("--article-path", type=str, help="Path to the input article text file.")
    parser.add_argument("--len", type=int, help="Desired number of tokens.")
    parser.add_argument("--out-folder", type=str, help="Path to save the truncated text.")

    args = parser.parse_args()

    # Load the article from the specified file
    with open(args.article_path, 'r') as file:
        article = file.read() 
        n_words = len(article.split(' '))
        if args.len > n_words: # If the article is short, expand it.
            print("Article tokens are not enough. Duplicate it.")
            article = article * math.ceil(args.len/n_words)
        # Ask the model to do something
        article = "Summarize the poem:\n" + article 
    

    # Truncate the article to the desired token count
    truncated_text = truncate_text_to_token_count(article, args.len)

    # Save the truncated text to the specified output file
    out_file = os.path.join(args.out_folder, str(args.len) + ".txt")
    with open(out_file, 'w') as output_file:
        output_file.write(truncated_text)

    print(f"Truncated text with {args.len} tokens saved to {out_file}.")

if __name__ == "__main__":
    main()

'''
python generate_datasets.py \
    --article-path /app/vllm/benchmarks/sonnet.txt --len 2500 \
    --out-folder /home/jacchang/POC_RFP/vllm/Mixtral-8x7B-Instruct-v0.1/ServerMode/Datasets/

python generate_datasets.py \
    --article-path /app/vllm/benchmarks/sonnet.txt --len 5500 \
    --out-folder /home/jacchang/POC_RFP/vllm/Mixtral-8x7B-Instruct-v0.1/ServerMode/Datasets/

python generate_datasets.py \
    --article-path /app/vllm/benchmarks/sonnet.txt --len 11000 \
    --out-folder /home/jacchang/POC_RFP/vllm/Mixtral-8x7B-Instruct-v0.1/ServerMode/Datasets/

'''