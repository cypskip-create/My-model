# AfriCode LM 🌍

**The African Developer’s AI Code Assistant**

AfriCode LM is a GPT-style language model built specifically for African developers, trained on African API documentation, mobile money integrations, USSD applications, and African fintech solutions.

-----

## Why AfriCode?

Most AI coding assistants are built for Western developers using Western infrastructure. African developers face unique challenges:

- **M-Pesa / Safaricom** STK Push integrations
- **Paystack, Flutterwave** payment processing
- **MTN MoMo, Airtel Money** mobile money APIs
- **USSD** application development
- **Low-bandwidth** optimized code patterns
- Local African fintech ecosystems

No major AI lab is focused on this. AfriCode fills that gap.

-----

## Project Structure

```
africode-lm/
├── model.py            # Transformer architecture (AfriCode LM)
├── train.py            # Training loop with African API code sampling
├── generate.py         # Code generation with African API templates
├── data_pipeline.py    # Data collection from African tech sources
├── africode_urls.txt   # African API documentation URLs
└── processed/          # Tokenized training data (generated)
└── checkpoints/        # Saved model weights (generated)
```

-----

## Quick Start

### 1. Install Dependencies

```bash
pip install torch tiktoken numpy datasets requests beautifulsoup4 tqdm
```

### 2. Collect Training Data

**Step 1 — Coding foundation (start here):**

```bash
# Python code from GitHub (3M+ files)
python data_pipeline.py \
  --source hf \
  --dataset bigcode/the-stack-smol \
  --dataset_config python \
  --output africode_python \
  --output_dir ./processed \
  --analyze
```

**Step 2 — JavaScript (for Node.js API integrations):**

```bash
python data_pipeline.py \
  --source hf \
  --dataset bigcode/the-stack-smol \
  --dataset_config javascript \
  --output africode_js \
  --output_dir ./processed
```

**Step 3 — African API documentation:**

```bash
python data_pipeline.py \
  --source africode \
  --output africode_apis \
  --output_dir ./processed \
  --analyze
```

**Step 4 — Your own collected data:**

```bash
# Drop .txt, .py, .js, .json files into ./raw_data/
python data_pipeline.py \
  --source files \
  --input_dir ./raw_data \
  --output africode_custom \
  --output_dir ./processed
```

### 3. Train the Model

```bash
python train.py \
  --data_train processed/africode_python_train.bin \
  --data_val processed/africode_python_val.bin \
  --max_steps 10000
```

### 4. Generate Code

```bash
# Use a built-in African API template
python generate.py --checkpoint checkpoints/best_model.pt --template mpesa
python generate.py --checkpoint checkpoints/best_model.pt --template paystack
python generate.py --checkpoint checkpoints/best_model.pt --template ussd

# Custom prompt
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "# How to integrate M-Pesa STK Push in Django REST Framework"

# Interactive mode
python generate.py --checkpoint checkpoints/best_model.pt --interactive
```

-----

## Training Data Strategy

### Recommended Datasets (in order of priority)

|Priority|Dataset               |Command                                                       |What it gives you            |
|--------|----------------------|--------------------------------------------------------------|-----------------------------|
|1       |The Stack (Python)    |`--dataset bigcode/the-stack-smol --dataset_config python`    |Strong coding foundation     |
|2       |The Stack (JavaScript)|`--dataset bigcode/the-stack-smol --dataset_config javascript`|Node.js/Express API skills   |
|3       |African API docs      |`--source africode`                                           |M-Pesa, Paystack, Flutterwave|
|4       |CodeSearchNet         |`--dataset code_search_net`                                   |Docstrings + code pairs      |
|5       |Your Q&A pairs        |`--source qa`                                                 |Custom fine-tuning           |

### Building Q&A Pairs for Fine-Tuning

Create JSON files in `./qa_pairs/` with this format:

```json
[
  {
    "question": "How do I integrate M-Pesa STK Push payment in Python?",
    "answer": "Here is a complete M-Pesa STK Push implementation:\n\nimport requests\nimport base64\nfrom datetime import datetime\n\ndef mpesa_stk_push(phone_number, amount):\n    ..."
  },
  {
    "question": "How do I accept Paystack payments in Django?",
    "answer": "To integrate Paystack in Django:\n\n1. Install: pip install paystack\n..."
  }
]
```

Then run:

```bash
python data_pipeline.py --source qa --input_dir ./qa_pairs --output africode_ft
python train.py --data_train processed/africode_ft_train.bin --data_val processed/africode_ft_val.bin
```

-----

## Model Sizes

|Config         |Params|Good for                    |GPU needed|
|---------------|------|----------------------------|----------|
|Small (default)|~25M  |Experiments, Colab free tier|T4 (free) |
|Medium         |~85M  |Serious training            |A100      |
|Large          |~300M |Production quality          |A100 x2   |

### Scale up:

```bash
# Medium (~85M params — GPT-2 equivalent)
python train.py \
  --data_train processed/africode_python_train.bin \
  --data_val processed/africode_python_val.bin \
  --d_model 768 --n_heads 12 --n_layers 12 --d_ff 3072 \
  --context_len 1024 --batch_size 8 --max_steps 50000
```

-----

## Training on Google Colab (Free)

```python
# Paste into a Colab cell — Runtime > Change runtime type > T4 GPU

!git clone https://github.com/YOUR_USERNAME/africode-lm.git
%cd africode-lm
!pip install torch tiktoken numpy datasets requests beautifulsoup4 tqdm

# Get coding data
!python data_pipeline.py \
    --source hf \
    --dataset bigcode/the-stack-smol \
    --dataset_config python \
    --output africode \
    --output_dir ./processed \
    --analyze

# Train
!python train.py \
    --data_train processed/africode_train.bin \
    --data_val processed/africode_val.bin \
    --max_steps 10000

# Generate code
!python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --template mpesa
```

-----

## Roadmap

- [x] Base transformer architecture
- [x] Data pipeline with African API sources
- [x] African API code generation templates
- [ ] Fine-tuning on Q&A pairs
- [ ] REST API wrapper
- [ ] Web interface for developers
- [ ] Fine-tune on M-Pesa, Paystack, Flutterwave docs
- [ ] Support for Swahili + English mixed prompts
- [ ] VSCode extension

-----

## Supported African APIs

|API               |Country             |Use Case                 |
|------------------|--------------------|-------------------------|
|M-Pesa (Safaricom)|Kenya               |Mobile payments, STK Push|
|Paystack          |Nigeria, Ghana, SA  |Online payments          |
|Flutterwave       |Pan-Africa          |Payments, transfers      |
|MTN MoMo          |17 African countries|Mobile money             |
|Airtel Money      |14 African countries|Mobile payments          |
|Chipper Cash      |Pan-Africa          |P2P transfers            |
|DPO Pay           |Pan-Africa          |Online payments          |

-----

*Built in Nairobi. For African developers, by African developers.*
