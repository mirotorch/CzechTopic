import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from .config import CrossEncoderConfig
from .model import TopicCrossEncoder

def smooth_highlights(binary_predictions):
    for i in range(1, len(binary_predictions) - 1):
        if binary_predictions[i] == 0:
            if binary_predictions[i-1] == 1 and binary_predictions[i+1] == 1:
                binary_predictions[i] = 1 
    return binary_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    base_config = AutoConfig.from_pretrained(model_name)
    config = CrossEncoderConfig(**base_config.to_dict())
    
    model = TopicCrossEncoder(config, technique="top3")
    
    checkpoint_path = "./output/best_model_top3.pt"
    print(f"Loading trained weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval()

    texts_csv_path = "./dataset/test-dataset/texts.csv"
    topics_csv_path = "./dataset/test-dataset/topics.csv"
    output_file = "./output/predictions.jsonl"
    
    print(f"Reading CSVs and grouping by cluster...")
    texts_df = pd.read_csv(texts_csv_path, sep='\t', encoding='utf-8')
    topics_df = pd.read_csv(topics_csv_path, sep='\t', encoding='utf-8')

    
    topics_df['topic_description'] = topics_df['topic_description'].fillna("")
    topics_df['topic_name'] = topics_df['topic_name'].fillna("")
    texts_df['text'] = texts_df['text'].fillna("")

    merged_df = pd.merge(texts_df, topics_df, on='cluster_id', suffixes=('_text', '_topic'))
    
    print(f"Generated {len(merged_df)} Text-Topic pairs. Starting inference...\n")

    with open(output_file, 'w', encoding='utf-8') as out_f:
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
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids']
                )
                scores = outputs["logits"]

            sep_idx = (inputs['input_ids'][0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()

            text_probs = scores[0, sep_idx + 1: -1].tolist()
            text_offsets = inputs['offset_mapping'][0, sep_idx + 1: -1].tolist()

            binary_preds = [1 if p >= 0.90 else 0 for p in text_probs]
            smoothed_preds = smooth_highlights(binary_preds)

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

    print("\nInference complete! Predictions saved to", output_file)

if __name__ == "__main__":
    main()