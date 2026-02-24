## “””
data_pipeline.py — Data Collection, Cleaning & Preparation Pipeline

This pipeline helps you:

1. Collect text from multiple sources (files, web, datasets)
1. Clean and normalize the text
1. Deduplicate
1. Tokenize and save as binary (.bin) for fast training
1. Analyze your dataset quality

Usage:
python data_pipeline.py –source files –input_dir ./raw_data –output data.bin
python data_pipeline.py –source web    –urls urls.txt       –output data.bin
python data_pipeline.py –source hf     –dataset wikitext    –output data.bin

Requirements:
pip install tiktoken datasets requests beautifulsoup4 tqdm numpy
“””

import os
import re
import sys
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Iterator

# —————————————————————————

# Argument Parsing

# —————————————————————————

def parse_args():
parser = argparse.ArgumentParser(description=“Data Pipeline for TransformerLM”)
parser.add_argument(”–source”,     type=str, required=True,
choices=[“files”, “web”, “hf”],
help=“Data source: local files, web scraping, or HuggingFace datasets”)
parser.add_argument(”–input_dir”,  type=str, default=”./raw_data”,
help=”[files] Directory containing .txt files”)
parser.add_argument(”–urls”,       type=str, default=“urls.txt”,
help=”[web] Text file with one URL per line”)
parser.add_argument(”–dataset”,    type=str, default=“wikitext”,
help=”[hf] HuggingFace dataset name”)
parser.add_argument(”–dataset_config”, type=str, default=“wikitext-103-raw-v1”,
help=”[hf] HuggingFace dataset config”)
parser.add_argument(”–output”,     type=str, default=“data.bin”,
help=“Output binary file for tokenized data”)
parser.add_argument(”–output_dir”, type=str, default=”./processed”,
help=“Directory for processed outputs”)
parser.add_argument(”–min_length”, type=int, default=100,
help=“Minimum characters per document”)
parser.add_argument(”–max_length”, type=int, default=100_000,
help=“Maximum characters per document”)
parser.add_argument(”–val_frac”,   type=float, default=0.05,
help=“Fraction of data to reserve for validation”)
parser.add_argument(”–analyze”,    action=“store_true”,
help=“Run dataset analysis after processing”)
return parser.parse_args()

# —————————————————————————

# Text Cleaning

# —————————————————————————

class TextCleaner:
“”“Cleans and normalizes raw text documents.”””

```
def __init__(self, min_length=100, max_length=100_000):
    self.min_length = min_length
    self.max_length = max_length

def clean(self, text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes and other control characters (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse excessive whitespace (but preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)           # multiple spaces → single
    text = re.sub(r"\n{3,}", "\n\n", text)         # 3+ newlines → 2

    # Remove lines that are just punctuation or numbers (often noise)
    lines = text.split("\n")
    lines = [l for l in lines if not re.match(r"^[\W\d]+$", l.strip()) or len(l.strip()) == 0]
    text = "\n".join(lines)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

def is_valid(self, text: str) -> bool:
    """Filter out documents that are too short, too long, or low quality."""
    if len(text) < self.min_length:
        return False
    if len(text) > self.max_length:
        # Truncate instead of discard
        return True

    # Filter documents with too little alphabetic content (e.g. tables, code noise)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False

    return True

def truncate(self, text: str) -> str:
    if len(text) > self.max_length:
        # Truncate at last sentence boundary before max_length
        truncated = text[:self.max_length]
        last_period = truncated.rfind(".")
        if last_period > self.max_length * 0.8:
            truncated = truncated[:last_period + 1]
        return truncated
    return text

def process(self, text: str):
    """Full processing pipeline for a single document."""
    text = self.clean(text)
    if not self.is_valid(text):
        return None
    text = self.truncate(text)
    return text
```

# —————————————————————————

# Deduplication

# —————————————————————————

class Deduplicator:
“”“Simple hash-based exact deduplication.”””

```
def __init__(self):
    self.seen = set()

def is_duplicate(self, text: str) -> bool:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    if h in self.seen:
        return True
    self.seen.add(h)
    return False

def reset(self):
    self.seen.clear()
```

# —————————————————————————

# Data Sources

# —————————————————————————

class FileSource:
“”“Load text from a directory of .txt files.”””

```
def __init__(self, input_dir: str):
    self.input_dir = Path(input_dir)
    self.files = list(self.input_dir.rglob("*.txt")) + \
                 list(self.input_dir.rglob("*.md"))  + \
                 list(self.input_dir.rglob("*.json"))

    if not self.files:
        raise FileNotFoundError(f"No .txt/.md/.json files found in {input_dir}")

    print(f"Found {len(self.files)} files in {input_dir}")

def iter_documents(self) -> Iterator[str]:
    for filepath in tqdm(self.files, desc="Reading files"):
        try:
            if filepath.suffix == ".json":
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                    # Handle common JSON formats
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                yield item
                            elif isinstance(item, dict):
                                for key in ["text", "content", "body", "document"]:
                                    if key in item:
                                        yield str(item[key])
                                        break
                    elif isinstance(data, dict):
                        for key in ["text", "content", "body"]:
                            if key in data:
                                yield str(data[key])
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    yield f.read()
        except Exception as e:
            print(f"  Warning: Could not read {filepath}: {e}")
```

class WebSource:
“”“Scrape text from a list of URLs.”””

```
def __init__(self, urls_file: str):
    try:
        import requests
        from bs4 import BeautifulSoup
        self.requests = requests
        self.BeautifulSoup = BeautifulSoup
    except ImportError:
        raise ImportError("Run: pip install requests beautifulsoup4")

    with open(urls_file, "r") as f:
        self.urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Loaded {len(self.urls)} URLs from {urls_file}")

def iter_documents(self) -> Iterator[str]:
    for url in tqdm(self.urls, desc="Scraping URLs"):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; DataPipeline/1.0)"}
            resp = self.requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            soup = self.BeautifulSoup(resp.text, "html.parser")

            # Remove nav, footer, scripts, ads
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            # Extract main content
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main:
                text = main.get_text(separator="\n")
                yield text

        except Exception as e:
            print(f"  Warning: Failed to scrape {url}: {e}")
```

class HuggingFaceSource:
“”“Load a dataset from HuggingFace datasets hub.”””

```
def __init__(self, dataset_name: str, config: str = None):
    try:
        from datasets import load_dataset
        self.load_dataset = load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"Loading HuggingFace dataset: {dataset_name} ({config})")
    self.dataset = load_dataset(dataset_name, config, trust_remote_code=True)
    self.dataset_name = dataset_name

def iter_documents(self) -> Iterator[str]:
    split = "train"
    if split not in self.dataset:
        split = list(self.dataset.keys())[0]

    data = self.dataset[split]
    text_keys = ["text", "content", "body", "document", "article"]

    for item in tqdm(data, desc=f"Loading {self.dataset_name}"):
        for key in text_keys:
            if key in item and item[key]:
                yield str(item[key])
                break
```

# —————————————————————————

# Tokenizer & Binary Serialization

# —————————————————————————

class DatasetBuilder:
“”“Tokenize documents and write to binary files for fast training.”””

```
def __init__(self, output_dir: str, val_frac: float = 0.05):
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.val_frac = val_frac

    try:
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        raise ImportError("Run: pip install tiktoken")

def build(self, documents: List[str], output_name: str = "data"):
    """Tokenize all documents and write train/val binary files."""
    print(f"\nTokenizing {len(documents):,} documents...")

    all_tokens = []
    total_chars = 0

    for doc in tqdm(documents, desc="Tokenizing"):
        tokens = self.enc.encode(doc, allowed_special={"<|endoftext|>"})
        # Add end-of-document token between documents
        tokens.append(self.enc.eot_token)
        all_tokens.extend(tokens)
        total_chars += len(doc)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    total_tokens = len(all_tokens)

    # Split into train/val
    split_idx = int(total_tokens * (1 - self.val_frac))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    # Save binary files
    train_path = self.output_dir / f"{output_name}_train.bin"
    val_path = self.output_dir / f"{output_name}_val.bin"

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    # Save metadata
    meta = {
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_chars": total_chars,
        "total_documents": len(documents),
        "vocab_size": self.enc.n_vocab,
        "encoding": "gpt2",
    }
    meta_path = self.output_dir / f"{output_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset built successfully!")
    print(f"  Total documents : {len(documents):,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total tokens    : {total_tokens:,}")
    print(f"  Train tokens    : {len(train_tokens):,}")
    print(f"  Val tokens      : {len(val_tokens):,}")
    print(f"  Train file      : {train_path}")
    print(f"  Val file        : {val_path}")
    print(f"  Metadata        : {meta_path}")
    print(f"{'='*50}\n")

    return str(train_path), str(val_path)
```

# —————————————————————————

# Dataset Analyzer

# —————————————————————————

def analyze_dataset(documents: List[str]):
“”“Print quality statistics about your dataset.”””
print(”\n— Dataset Analysis —”)

```
lengths = [len(d) for d in documents]
total_chars = sum(lengths)
avg_len = total_chars / max(len(documents), 1)

print(f"Total documents   : {len(documents):,}")
print(f"Total characters  : {total_chars:,}")
print(f"Avg doc length    : {avg_len:,.0f} chars")
print(f"Min doc length    : {min(lengths):,} chars")
print(f"Max doc length    : {max(lengths):,} chars")

# Word count estimate
total_words = sum(len(d.split()) for d in documents)
print(f"Estimated words   : {total_words:,}")
print(f"Estimated tokens  : ~{int(total_words * 1.3):,}  (words × 1.3)")

# Quality check
alpha_ratios = [sum(c.isalpha() for c in d) / max(len(d), 1) for d in documents]
avg_alpha = sum(alpha_ratios) / max(len(alpha_ratios), 1)
print(f"Avg alpha ratio   : {avg_alpha:.2%}  (higher = more text, less noise)")

# Size guidance
token_estimate = int(total_words * 1.3)
print(f"\n--- Training Guidance ---")
if token_estimate < 1_000_000:
    print(f"⚠️  Small dataset ({token_estimate:,} tokens). Model may overfit quickly.")
    print(f"   Tip: Try to gather at least 10M tokens for meaningful training.")
elif token_estimate < 10_000_000:
    print(f"✓  Medium dataset ({token_estimate:,} tokens). Good for small models (3–10M params).")
elif token_estimate < 100_000_000:
    print(f"✓  Large dataset ({token_estimate:,} tokens). Good for medium models (25–85M params).")
else:
    print(f"🚀 Very large dataset ({token_estimate:,} tokens). Ready to train serious models!")

print()
```

# —————————————————————————

# Main

# —————————————————————————

def main():
args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

```
# Initialize cleaner and deduplicator
cleaner = TextCleaner(min_length=args.min_length, max_length=args.max_length)
deduper = Deduplicator()

# Select source
if args.source == "files":
    source = FileSource(args.input_dir)
elif args.source == "web":
    source = WebSource(args.urls)
elif args.source == "hf":
    source = HuggingFaceSource(args.dataset, args.dataset_config)

# Process documents
documents = []
stats = {"total": 0, "too_short": 0, "low_quality": 0, "duplicate": 0, "accepted": 0}

print("\nProcessing documents...")
for raw_doc in source.iter_documents():
    stats["total"] += 1
    cleaned = cleaner.process(raw_doc)

    if cleaned is None:
        stats["low_quality"] += 1
        continue

    if deduper.is_duplicate(cleaned):
        stats["duplicate"] += 1
        continue

    documents.append(cleaned)
    stats["accepted"] += 1

# Stats
print(f"\n--- Processing Stats ---")
print(f"  Total raw docs  : {stats['total']:,}")
print(f"  Low quality     : {stats['low_quality']:,}")
print(f"  Duplicates      : {stats['duplicate']:,}")
print(f"  Accepted        : {stats['accepted']:,}")

if not documents:
    print("ERROR: No documents passed the filter. Check your data source or lower --min_length.")
    sys.exit(1)

# Analyze if requested
if args.analyze:
    analyze_dataset(documents)

# Build tokenized dataset
builder = DatasetBuilder(output_dir=args.output_dir, val_frac=args.val_frac)
output_name = Path(args.output).stem
train_path, val_path = builder.build(documents, output_name=output_name)

print(f"✓ Pipeline complete!")
print(f"\nTo train your model with this data, update train.py to load binary files:")
print(f"    train_path = '{train_path}'")
print(f"    val_path   = '{val_path}'")
print(f"\nOr run: python train.py --data_train {train_path} --data_val {val_path}")
```

if **name** == “**main**”:
main()
