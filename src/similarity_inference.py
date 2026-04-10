import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModel

class BaselineDotProductMatcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e_text, e_topic):
        e_text_norm = F.normalize(e_text, p=2, dim=-1)
        e_topic_norm = F.normalize(e_topic, p=2, dim=-1)
        similarity_matrix = torch.matmul(e_text_norm, e_topic_norm.transpose(1, 2))
        
        k = min(3, similarity_matrix.size(2))
        top_k_values = torch.topk(similarity_matrix, k=k, dim=2).values
        sim_agg = top_k_values.mean(dim=2)
        
        return sim_agg


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
    
    model = AutoModel.from_pretrained(model_name)
    
    raw_state_dict = torch.load("./output/best_model.pt", map_location=device)
    clean_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith('bert.'):
            clean_state_dict[key[5:]] = value
        elif key.startswith('roberta.'):
            clean_state_dict[key[8:]] = value
        else:
            clean_state_dict[key] = value

    model.load_state_dict(clean_state_dict, strict=False) 
    model.to(device)
    model.eval()
    matcher = BaselineDotProductMatcher().to(device)

    input_file = "./dataset/test-dataset/0.jsonl"
    output_file = "./predictions.jsonl"
    
    print(f" Reading {input_file} and saving to {output_file}...\n")

    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        for line in f:
            data = json.loads(line)
            
            name = data.get('topic_name', '')
            description = data.get('topic_description', '')
            topic = f"{name} {description}".strip()
            text = data.get('text', '')

            inputs = tokenizer(
                topic, 
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                return_offsets_mapping=True 
            ).to(device)

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                vectors = outputs.last_hidden_state

            sep_idx = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()

            e_topic = vectors[:, 1:sep_idx, :] 
            e_text = vectors[:, sep_idx + 1: -1, :] 

            predictions = matcher(e_text, e_topic)
            text_probs = predictions[0].tolist()
            
            text_offsets = inputs['offset_mapping'][0, sep_idx + 1: -1].tolist()

            binary_preds = [1 if p >= 0.85 else 0 for p in text_probs]
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
                "cluster_id": data.get("cluster_id", 0),
                "annotations": annotations,
                "text_id": data.get("text_id", 0),
                "topic_id": data.get("topic_id", 0)
            }

            out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()