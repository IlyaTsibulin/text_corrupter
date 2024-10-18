import random
import numpy as np
import nltk
import argparse
import re
import jsonlines
from tqdm import tqdm

nltk.download('punkt')
from nltk.tokenize import word_tokenize

def corrupt_text_with_extra_ids(text, corruption_rate, mean_span_length, num_spans):
    tokens = word_tokenize(text)
    L = len(tokens)
    
    total_corrupt_tokens = int(L * corruption_rate)
    
    spans = []
    for _ in range(num_spans):
        span_length = max(1, int(np.random.normal(mean_span_length, 1)))  
        span_start = random.randint(0, L - span_length)  
        span_end = span_start + span_length
        spans.append((span_start, span_end))
    
    spans = sorted(spans, key=lambda x: x[0])
    
    corrupted_tokens = tokens[:]
    extra_id_counter = 0
    mask_indices = []
    
    current_position = 0
    token_positions = []
    
    for token in tokens:
        token_start = text.find(token, current_position)
        token_end = token_start + len(token)
        token_positions.append((token_start, token_end))
        current_position = token_end  
    
    for span_start, span_end in spans:
        corrupted_tokens[span_start:span_end] = [f'<extra_id_{extra_id_counter}>']
        start_index = token_positions[span_start][0]  
        end_index = token_positions[span_end - 1][1] 
        mask_indices.append((start_index, end_index))
        extra_id_counter += 1

    corrupted_text = ' '.join(corrupted_tokens)

    corrupted_text = re.sub(r'\s+([,.!?;:])', r'\1', corrupted_text)  
    corrupted_text = re.sub(r'([,.!?;:])(?=[^\s])', r'\1 ', corrupted_text)  

    corrupted_text = re.sub(r'\(\s*', '(', corrupted_text) 
    corrupted_text = re.sub(r'\s*\)', ')', corrupted_text) 
    corrupted_text = re.sub(r'\[\s*', '[', corrupted_text)
    corrupted_text = re.sub(r'\s*\]', ']', corrupted_text)
    corrupted_text = re.sub(r'«\s*', '«', corrupted_text)  
    corrupted_text = re.sub(r'\s*»', '»', corrupted_text)  

    corrupted_text = re.sub(r'([,.!?;:])\s+([,.!?;:])', r'\1\2', corrupted_text)  

    return corrupted_text, mask_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Коррупция текста с использованием extra_id токенов")
    
    parser.add_argument("--corruption_rate", type=float, default=0.3, help="Процент текста, который будет испорчен")
    parser.add_argument("--mean_span_length", type=int, default=3, help="Средняя длина поврежденных фрагментов")
    parser.add_argument("--num_spans", type=int, default=3, help="Количество фрагментов для порчи")
    parser.add_argument("--input_file", type=str, required=True, help="Путь к входному файлу output.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Путь к выходному файлу для сохранения результатов")

    args = parser.parse_args()

    with jsonlines.open(args.input_file) as reader, jsonlines.open(args.output_file, mode='w') as writer:
        for obj in tqdm(reader, desc="Processing texts", unit="text"):
            if 'text' in obj:
                text = obj['text']
                corrupted_text, mask_indices = corrupt_text_with_extra_ids(
                    text, args.corruption_rate, args.mean_span_length, args.num_spans
                )
                writer.write({
                    'original_text': text,
                    'corrupted_text': corrupted_text,
                    'mask_indices': mask_indices
                })

    print(f"Corrupted texts written to {args.output_file}")
