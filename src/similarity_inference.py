import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import argparse

from .checkpoints import load_checkpoint
from .config import CrossEncoderConfig
from .model import TopicCrossEncoder
from .predict import span_predictions_to_spans

def expand_subword_highlights(binary_preds, tokens):
    fixed_preds = list(binary_preds)
    for i in range(1, len(tokens)):
        if tokens[i].startswith("##") and fixed_preds[i-1] == 1:
            fixed_preds[i] = 1
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i+1].startswith("##") and fixed_preds[i+1] == 1:
            fixed_preds[i] = 1
    return fixed_preds

def smooth_highlights(binary_predictions):
    for i in range(1, len(binary_predictions) - 1):
        if binary_predictions[i] == 0:
            if binary_predictions[i-1] == 1 and binary_predictions[i+1] == 1:
                binary_predictions[i] = 1 
    return binary_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ufal/robeczech-base")
    parser.add_argument("--model-path", type=str, default="./output/best_model_top3.pt")
    parser.add_argument("--texts-csv", type=str, default="./dataset/test-dataset/texts.csv")
    parser.add_argument("--topics-csv", type=str, default="./dataset/test-dataset/topics.csv")
    parser.add_argument("--output-file", type=str, default="./output/predictions.jsonl")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--technique", type=str, default="top3")
    parser.add_argument("--max-span-length", type=int, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.sep_token_id = tokenizer.sep_token_id
    
    checkpoint = load_checkpoint(args.model_path, map_location=device)
    max_span_length = (
        args.max_span_length
        if args.max_span_length is not None
        else checkpoint.get("max_span_length", 4)
    )
    model = TopicCrossEncoder(
        config,
        technique=args.technique,
        max_span_length=max_span_length,
    )
    
    print(f"Loading trained weights from {args.model_path}...")
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = (
        args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.90)
    )
    
    model.to(device)
    model.eval()
    print(f"Using threshold: {threshold:.2f}")
    if args.technique in {"span", "span-max"}:
        print(f"Using max span length: {max_span_length}")

    print(f"Reading CSVs and grouping by cluster...")
    texts_df = pd.read_csv(args.texts_csv, sep='\t', encoding='utf-8')
    topics_df = pd.read_csv(args.topics_csv, sep='\t', encoding='utf-8')

    topics_df['topic_description'] = topics_df['topic_description'].fillna("")
    topics_df['topic_name'] = topics_df['topic_name'].fillna("")
    texts_df['text'] = texts_df['text'].fillna("")

    merged_df = pd.merge(texts_df, topics_df, on='cluster_id', suffixes=('_text', '_topic'))
    
    print(f"Generated {len(merged_df)} Text-Topic pairs. Starting inference...\n")

    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for index, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            
            name = str(row['topic_name'])
            description = str(row['topic_description'])
            topic = f"{name} {description}".strip()
            text = str(row['text'])

            inputs = tokenizer(
                topic, 
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                return_offsets_mapping=True 
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask']
                )
                scores = outputs["logits"]

            if args.technique in {"span", "span-max"}:
                annotations = span_predictions_to_spans(
                    text,
                    outputs["span_scores"][0],
                    outputs["span_indices"][0],
                    inputs["offset_mapping"][0],
                    threshold=threshold,
                )
            else:
                sep_positions = (inputs['input_ids'][0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                
                if len(sep_positions) >= 2:
                    first_sep = sep_positions[0].item()
                    second_sep = sep_positions[1].item()
                    text_probs = scores[0, first_sep + 1: second_sep].tolist()
                    text_offsets = inputs['offset_mapping'][0, first_sep + 1: second_sep].tolist()
                    text_ids = inputs['input_ids'][0, first_sep + 1: second_sep].tolist()
                elif len(sep_positions) == 1:
                    first_sep = sep_positions[0].item()
                    text_probs = scores[0, first_sep + 1: -1].tolist()
                    text_offsets = inputs['offset_mapping'][0, first_sep + 1: -1].tolist()
                    text_ids = inputs['input_ids'][0, first_sep + 1: -1].tolist()
                else:
                    text_probs = scores[0].tolist()
                    text_offsets = inputs['offset_mapping'][0].tolist()
                    text_ids = inputs['input_ids'][0].tolist()

                text_tokens = tokenizer.convert_ids_to_tokens(text_ids)
                binary_preds = [1 if p >= threshold else 0 for p in text_probs]
                
                expanded_preds = expand_subword_highlights(binary_preds, text_tokens)
                smoothed_preds = smooth_highlights(expanded_preds)

                annotations = []
                in_span = False
                current_start = None
                current_end = None

                for idx, is_highlight in enumerate(smoothed_preds):
                    start_char, end_char = text_offsets[idx]
                    
                    if start_char == 0 and end_char == 0:
                        continue
                        
                    if is_highlight == 1:
                        if not in_span:
                            in_span = True
                            current_start = start_char
                        current_end = end_char 
                    else:
                        if in_span:
                            in_span = False
                            annotations.append({
                                "start": current_start,
                                "end": current_end,
                                "text_piece": text[current_start:current_end]
                            })

                if in_span:
                    annotations.append({
                        "start": current_start,
                        "end": current_end,
                        "text_piece": text[current_start:current_end]
                    })

            output_obj = {
                "text": text,
                "topic_name": name,
                "topic_description": description,
                "cluster_id": int(row['cluster_id']),
                "annotations": annotations,
                "text_id": int(row.get('id_text', 0)),
                "topic_id": int(row.get('id_topic', 0))
            }

            out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

    print("\nInference complete! Predictions saved to", args.output_file)

if __name__ == "__main__":
    main()
