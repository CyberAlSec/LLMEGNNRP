Usage

All examples assume the virtual environment is activated and all required packages are installed. 

Due to storage limitations, only the processed Cora dataset is currently available in the data/ directory. Processed versions of other datasets will be uploaded to Google Drive upon
publication. Unzip `gpt_responses.zip` in the `data/TAPE/` directory.

The TAPE results enhanced with LLM features are stored in `data/TAPE/` (please unzip `gpt_responses.zip`).
The attack results are in `attack/data/TAPE/`, and KEA-enhanced results are in `attack/data/KEA/`.

# 1) Launch Structural Attacks (DeepRobust)

Generate perturbed graph structure (example: mettack on cora):

```python
python attack/Structural.py --dataset cora --attack_method mettack 
```


# 2) Launch Textual Attacks (OpenAttack)

Typical workflow:

1. Train a surrogate text classifier (e.g., E5-Large) on node textual attributes. We provide `augmodel/trainLM.py` for this purpose:
```python
python augmodel/trainLM.py --dataset cora --feature_type E5 --surrogate True
```
2. Generate perturbed texts:
```python
# DeepWordBug & BERT-Attack:
python attack/Textual.py --dataset cora --text_attack DWord --feature_type E5

# MAYA attack:
python attack/MAYA/attack.py --dataset cora --feature_type E5
```

# 3) Generate LLM/LM-driven Features
All generated LLM/LM features (i.e., LLM/LM-enhanced node embeddings) are saved in `data/emb/`. Example commands:

## TAPE: 
```python
# Generate LLM-enhanced attributes
python LLM_Utils.py --dataset cora --feature_type TAPE [--text_attack DWord]

# Generate features (two modes)
python augmodel/trainLM.py --dataset cora --feature_type TAPE [--text_attack DWord]
python augmodel/trainLM.py --dataset cora --feature_type TAPE --lm_use_gpt True [--text_attack DWord]
```

## KEA:
```python
# Generate LLM-enhanced attributes
python LLM_Utils.py --dataset cora --feature_type KEA [--text_attack DWord]

# Generate features
python gen_KEA.py --dataset cora [--text_attack DWord]
```

## TE3L:
```python
# Without textual attack:
python augmodel/Api_emb.py --dataset cora

# Under textual attack:
python augmodel/Api_emb.py --dataset cora --text_attack DWord
```

## LLaMA:
```python
python augmodel/LLM_emb.py --dataset cora --model_type Llama

# Under textual attack:
python augmodel/LLM_emb.py --dataset cora --model_type Llama --style DWord
```

## Linq:
```python
python augmodel/LLM_emb.py --dataset cora --model_type Linq

# Under textual attack:
python augmodel/LLM_emb.py --dataset cora --model_type Linq --style DWord
```

## SimTeg:
```python
python augmodel/SimTeg/main.py --dataset cora [--text_attack DWord]
```

## E5:
```python
python augmodel/trainLM.py --dataset cora --feature_type E5

# Under textual attack:
python augmodel/trainLM.py --dataset cora --feature_type E5 --text_attack DWord
```

## ModernBert:
```python
python augmodel/trainLM.py --dataset cora --feature_type ModernBert

# Under textual attack:
python augmodel/trainLM.py --dataset cora --feature_type ModernBert --text_attack DWord
```

# 4) Train/Evaluate GNNs
# Examples for training/evaluating GNNs under different attack types:
```python
# Evaluate under textual attack( example feature type : ModernBert )
python main.py --dataset cora --feature_type ModernBert --text_attack DWord

# Evaluate under structural attack
python main.py --dataset cora --feature_type ModernBert --attack_method mettack

# Compute additional metrics / analysis
python Metric.py --dataset cora --text_attack DWord
```

