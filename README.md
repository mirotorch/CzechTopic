# CzechTopic

`CzechTopic` is a Czech topic-localization project built around a transformer cross-encoder. The model takes a topic name plus topic description together with a text, computes token-level similarities between the topic segment and the text segment, and predicts which parts of the text support the given topic.

The repository currently contains training, inference, and evaluation utilities, multiple pooling strategies for aggregating topic-to-text token similarities, prepared dataset splits, saved model checkpoints, and experiment logs from Weights & Biases.

## What the project does

- Trains a token-level topic localization model on CzechTopic data.
- Supports multiple similarity aggregation techniques: `max`, `mean`, `top3`, `top5`, `conv2`, and `conv3`.
- Runs inference on TSV/CSV-style test inputs and exports predictions as JSONL.
- Evaluates predictions against annotation files and computes topic-level metrics.
- Stores trained checkpoints and experiment metadata locally.

## How it works

The core model is implemented in `src/model.py` as a cross-encoder on top of Hugging Face transformers. Each input is encoded as:

- BERT-style: `[CLS] topic_name topic_description [SEP] text [SEP]`
- RoBERTa-style: `<s> topic_name topic_description </s></s> text </s>`

The model:

1. Encodes the combined sequence with a pretrained transformer.
2. Splits hidden states into topic tokens and text tokens using separator positions.
3. Computes normalized token-to-token similarity.
4. Aggregates topic-to-text similarity using one of the supported pooling techniques.
5. Produces token-level highlight probabilities for the text segment.

## Main files

- `src/train.py`: training loop, validation, threshold search, checkpoint saving, and W&B logging.
- `src/similarity_inference.py`: inference script that reads test CSV files, predicts highlights, and writes JSONL output.
- `src/run_evaluation.py`: runs model inference on JSONL test samples.
- `evaluation/evaluate.py`: evaluates predictions against multiple raters in the CzechTopic test dataset.
- `evaluation/evaluate_single.py`: evaluates one prediction file against one ground-truth JSONL file.

## Environment

- Python: `3.14.3`
- Local environment file added: `requirements.txt`
- Installed environment captured from: `venv\Scripts\python.exe -m pip freeze`

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python -m src.train --data-dir dataset/dev-dataset --output-dir output --model-name bert-base-multilingual-cased --technique max
```

Run similarity-based inference:

```bash
python -m src.similarity_inference --model-name ufal/robeczech-base --model-path ./output/best_model_top3_robeczech.pt --texts-csv ./dataset/test-dataset/texts.csv --topics-csv ./dataset/test-dataset/topics.csv --output-file ./output/predictions.jsonl --threshold 0.90 --technique top3
```

Run evaluation inference:

```bash
python -m src.run_evaluation --model-path output/best_model_top3_mbert.pt --model-name bert-base-multilingual-cased --data-dir dataset/test-dataset --output-path output/predictions.jsonl --threshold 0.5 --technique top3
```

Evaluate predictions:

```bash
python evaluation/evaluate.py --pred output/predictions.jsonl --gt dataset
```

## Project structure

The following structure reflects the current repository contents in your filesystem:

```text
.
    .gitignore
    README.md
    requirements.txt
    dataset
        .DS_Store
        dev-dataset
            texts.csv
            topics.csv
            train.jsonl
            val.jsonl
        test-dataset
            0.jsonl
            3.jsonl
            4.jsonl
            5.jsonl
            6.jsonl
            7.jsonl
            annotators.csv
            texts.csv
            topics.csv
        test-model-results
            ...
    evaluation
        common.py
        evaluate.py
        evaluate_single.py
    local
        ColBERT.png
    output
        best_model_baseline.pt
        best_model_conv2_jhu-clsp_mmBERT-base.pt
        best_model_top3_jhu-clsp-mmBERT-base.pt
        best_model_top3_mbert.pt
        best_model_top3_robeczech.pt
        predictions.jsonl
    src
        __init__.py
        collate.py
        config.py
        model.py
        predict.py
        run_evaluation.py
        similarity_inference.py
        techniques.py
        tokenizer.py
        train.py
        trainer.py
```

## Dependencies in local environment

The `requirements.txt` file was generated from the packages installed in your local virtual environment, so it reflects the exact versions currently available there.
