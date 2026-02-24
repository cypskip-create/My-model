# TransformerLM — Full Pipeline Guide

A complete, end-to-end pipeline for building and training your own language model.

-----

## Project Structure

```
your-ai-repo/
├── model.py            # Transformer architecture
├── train.py            # Training loop (updated — supports .bin files)
├── generate.py         # Text generation / inference
├── data_pipeline.py    # Data collection, cleaning & tokenization
├── urls.txt            # URLs to scrape (if using web source)
├── raw_data/           # Drop your .txt files here
└── processed/          # Pipeline outputs (tokenized .bin files)
```

-----

## Step 1: Install Dependencies

```bash
pip install torch tiktoken numpy datasets requests beautifulsoup4 tqdm
```

-----

## Step 2: Collect & Prepare Your Data

You have 3 ways to get training data:

### Option A — Your Own Text Files (easiest)

Drop `.txt`, `.md`, or `.json` files into a `raw_data/` folder:

```bash
mkdir raw_data
# Copy your text files in
cp my_documents/*.txt raw_data/

python data_pipeline.py \
  --source files \
  --input_dir ./raw_data \
  --output data \
  --output_dir ./processed \
  --analyze
```

### Option B — Scrape Websites

Edit `urls.txt` and add one URL per line:

```bash
python data_pipeline.py \
  --source web \
  --urls urls.txt \
  --output data \
  --output_dir ./processed \
  --analyze
```

### Option C — HuggingFace Datasets (best for large scale)

Use any public dataset from huggingface.co/datasets:

```bash
# Wikipedia
python data_pipeline.py \
  --source hf \
  --dataset wikipedia \
  --dataset_config 20220301.en \
  --output wikipedia \
  --output_dir ./processed

# General web text
python data_pipeline.py \
  --source hf \
  --dataset wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --output wikitext \
  --output_dir ./processed

# Books
python data_pipeline.py \
  --source hf \
  --dataset bookcorpus \
  --output books \
  --output_dir ./processed
```

### Recommended Free Datasets by Domain

|Domain |Dataset            |HuggingFace Name        |
|-------|-------------------|------------------------|
|General|Wikipedia          |`wikipedia`             |
|General|WebText            |`Skylion007/openwebtext`|
|Books  |BookCorpus         |`bookcorpus`            |
|Code   |The Stack          |`bigcode/the-stack-smol`|
|Science|PubMed             |`pubmed_abstracts`      |
|Legal  |FreeLaw            |`free_law`              |
|Finance|FinancialPhraseBank|`financial_phrasebank`  |
|News   |CC-News            |`cc_news`               |

-----

## Step 3: Train Your Model

### From pre-tokenized binary files (recommended — much faster):

```bash
python train.py \
  --data_train processed/data_train.bin \
  --data_val processed/data_val.bin \
  --max_steps 10000
```

### From a raw text file:

```bash
python train.py --data your_text.txt --max_steps 5000
```

### Scale up model size:

```bash
# Medium model (~25M params)
python train.py \
  --data_train processed/data_train.bin \
  --data_val processed/data_val.bin \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 8 \
  --d_ff 2048 \
  --context_len 512 \
  --batch_size 16 \
  --max_steps 50000

# Large model (~85M params — GPT-2 equivalent)
python train.py \
  --data_train processed/data_train.bin \
  --data_val processed/data_val.bin \
  --d_model 768 \
  --n_heads 12 \
  --n_layers 12 \
  --d_ff 3072 \
  --context_len 1024 \
  --batch_size 8 \
  --max_steps 100000
```

### On Google Colab (free GPU):

```python
# Run in a Colab cell
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install torch tiktoken numpy datasets requests beautifulsoup4 tqdm

# Prepare data
!python data_pipeline.py --source hf --dataset wikitext \
    --dataset_config wikitext-103-raw-v1 --output wikitext --output_dir ./processed

# Train
!python train.py \
    --data_train processed/wikitext_train.bin \
    --data_val processed/wikitext_val.bin \
    --max_steps 5000
```

-----

## Step 4: Generate Text

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "Artificial intelligence is" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 50 \
  --num_samples 3
```

-----

## Understanding Your Training Output

```
Step   50 | loss 4.2310 | lr 3.00e-04 | 12,450 tok/s
Step  100 | loss 3.8821 | lr 3.00e-04 | 12,512 tok/s

[Eval step 500] train=3.1204  val=3.2891
✓ New best val loss! Saved to checkpoints/best_model.pt

--- Sample at step 500 ---
The quick brown fox jumped over the lazy...
```

|Metric    |What it means                                      |
|----------|---------------------------------------------------|
|`loss`    |Lower = better. Starts ~4-5, good models reach ~2-3|
|`val loss`|Validation loss — key indicator of real learning   |
|`tok/s`   |Training speed in tokens per second                |

**Loss interpretation:**

- `loss > 4.0` — Early training, model is barely learning
- `loss 3.0–4.0` — Model learning basic patterns
- `loss 2.0–3.0` — Coherent text, domain knowledge emerging
- `loss < 2.0` — Strong model (requires lots of data + compute)

-----

## Resuming Training

```bash
python train.py \
  --data_train processed/data_train.bin \
  --data_val processed/data_val.bin \
  --resume checkpoints/ckpt_step5000.pt \
  --max_steps 20000
```

-----

## Hardware & Cost Estimates

|Model Size|Params|GPU          |Train Time (10K steps)|Cloud Cost|
|----------|------|-------------|----------------------|----------|
|Small     |10M   |Free Colab T4|~30 min               |Free      |
|Medium    |25M   |RTX 3090     |~1 hr                 |~$0.40    |
|Large     |85M   |RTX 4090     |~3 hrs                |~$1.50    |
|GPT-2     |117M  |A100         |~6 hrs                |~$6.00    |

Recommended cloud platforms (cheapest first):

1. **Google Colab** — Free T4/A100 (limited hours)
1. **Kaggle** — Free T4 (30 hrs/week)
1. **Vast.ai** — ~$0.20–0.40/hr
1. **RunPod** — ~$0.25–0.50/hr
1. **Lambda Labs** — ~$0.50–1.00/hr
