# V-ERR

A GRPO-based post-training method that identifies modality-specialized attention pathways and recurrently reinforces evidence-checking to counteract visual-signal fading, improving VLM multimodal reasoning.

## Overview

V-ERR enhances vision-language models by analyzing attention patterns during generation and dynamically adjusting token-level rewards. It identifies visual-specialized attention heads and reinforces tokens that actively reference key visual evidence.

## Features

- **Visual Pathways Identification**: Automatically detects attention heads specialized for visual information
- **Dynamic Token Weighting**: Adjusts rewards based on visual evidence utilization

## Installation

```bash
pip install -r requirements.txt
```

For model-specific dependencies, refer to their official repositories or huggingface page.

## Project Structure

```
RPER/
├── LLaVA-1.5-7B/
│   ├── base.py          # Standard GRPO
│   ├── rhythm.py        # Rhythm-aware GRPO
│   ├── verr.py          # VERR
│   ├── eval.py          # Evaluation script
│   └── utils.py         # Utility functions
├── Qwen2.5-VL-3B-Instruct/
├── Qwen3-VL-2B-Instruct/
└── datasets/            # Dataset directory
```

## Usage

### 1. Prepare Data

Organize your dataset in the following structure:
```
datasets/YOUR_DATASET/
├── train.json
├── test.json
└── images/
```

Dataset format:
```json
{
  "sample_id": {
    "question": "Question text",
    "choices": ["A", "B", "C", "D"],
    "answer": "A",
    "image": "images/sample.jpg"
  }
}
```

### 2. Configure Training

Edit the training script (e.g., `LLaVA-1.5-7B/rper.py`):
```python
model_path = "path/to/your/model"
output_dir = "path/to/output"
data_dir = "datasets/YOUR_DATASET"
```

### 3. Run Training

**Base GRPO:**
```bash
python LLaVA-1.5-7B/base.py
```

**Rhythm-aware GRPO:**
```bash
python LLaVA-1.5-7B/rhythm.py
```

**V-ERR:**
```bash
python LLaVA-1.5-7B/verr.py
```

### 4. Evaluation

```bash
python LLaVA-1.5-7B/eval.py
```
