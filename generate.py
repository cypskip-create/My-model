## “””
generate.py - AfriCode LM Code Generation

Load a trained AfriCode model and generate code for African API integrations.

Usage:
# Generate M-Pesa integration code
python generate.py –checkpoint checkpoints/best_model.pt   
–prompt “# How to integrate M-Pesa STK Push in Python”

```
# Generate Paystack code
python generate.py --checkpoint checkpoints/best_model.pt \
                   --prompt "# Paystack payment integration in Django"

# Interactive mode
python generate.py --checkpoint checkpoints/best_model.pt --interactive
```

Requirements:
pip install torch tiktoken
“””

import argparse
import torch
from model import ModelConfig, TransformerLM

# —————————————————————————

# African API Prompt Templates

# —————————————————————————

PROMPT_TEMPLATES = {
“mpesa”:       “# M-Pesa STK Push integration in Python\nimport requests\n\ndef mpesa_stk_push(”,
“paystack”:    “# Paystack payment integration\n# Initialize transaction\nimport requests\n\nPAYSTACK_SECRET = “,
“flutterwave”: “# Flutterwave payment integration in Python\nimport requests\n\ndef initiate_payment(”,
“mtn_momo”:    “# MTN Mobile Money API integration\nimport requests\n\nMTN_API_KEY = “,
“ussd”:        “# USSD menu implementation\n# Session handler\ndef handle_ussd_request(session_id, phone_number, text):\n”,
“airtel”:      “# Airtel Money API integration\nimport requests\n\ndef airtel_payment(”,
}

def parse_args():
parser = argparse.ArgumentParser(description=“AfriCode LM - Code Generation”)
parser.add_argument(”–checkpoint”,     type=str,   required=True,    help=“Path to model checkpoint”)
parser.add_argument(”–prompt”,         type=str,   default=None,     help=“Custom prompt text”)
parser.add_argument(”–template”,       type=str,   default=None,
choices=list(PROMPT_TEMPLATES.keys()),
help=“Use a built-in African API template”)
parser.add_argument(”–max_new_tokens”, type=int,   default=300,      help=“Max tokens to generate”)
parser.add_argument(”–temperature”,    type=float, default=0.7,      help=“Lower = more precise code”)
parser.add_argument(”–top_k”,          type=int,   default=40,       help=“Top-k sampling”)
parser.add_argument(”–num_samples”,    type=int,   default=1,        help=“Number of completions”)
parser.add_argument(”–interactive”,    action=“store_true”,          help=“Interactive prompt mode”)
return parser.parse_args()

def load_model(checkpoint_path: str, device: str):
“”“Load model from checkpoint.”””
ckpt  = torch.load(checkpoint_path, map_location=device)
cfg   = ModelConfig(**ckpt[“cfg”])
model = TransformerLM(cfg).to(device)
model.load_state_dict(ckpt[“model”])
model.eval()
print(f”AfriCode LM loaded | {model.num_parameters():,} params | step {ckpt.get(‘step’, ‘?’)}”)
return model, cfg

def generate_code(model, enc, prompt: str, device: str,
max_new_tokens=300, temperature=0.7, top_k=40) -> str:
“”“Generate code from a prompt.”””
tokens  = enc.encode(prompt)
idx     = torch.tensor([tokens], dtype=torch.long, device=device)
out     = model.generate(idx, max_new_tokens=max_new_tokens,
temperature=temperature, top_k=top_k)
return enc.decode(out[0].tolist())

def interactive_mode(model, enc, device, args):
“”“Interactive code generation loop.”””
print(”\nAfriCode LM - Interactive Mode”)
print(“Type your prompt and press Enter. Type ‘quit’ to exit.”)
print(“Available templates:”, “, “.join(PROMPT_TEMPLATES.keys()))
print(”=”*50)

```
while True:
    try:
        user_input = input("\nPrompt (or template name): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting AfriCode.")
        break

    if user_input.lower() in ("quit", "exit", "q"):
        break

    # Check if it's a template name
    if user_input.lower() in PROMPT_TEMPLATES:
        prompt = PROMPT_TEMPLATES[user_input.lower()]
        print(f"\nUsing template: {user_input}")
    else:
        prompt = user_input

    print(f"\n{'='*50}")
    print("Generated Code:")
    print('='*50)
    result = generate_code(model, enc, prompt, device,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature,
                           top_k=args.top_k)
    print(result)
    print('='*50)
```

def main():
args = parse_args()

```
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
except ImportError:
    raise ImportError("Run: pip install tiktoken")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model, cfg = load_model(args.checkpoint, device)

# Interactive mode
if args.interactive:
    interactive_mode(model, enc, device, args)
    return

# Determine prompt
if args.template:
    prompt = PROMPT_TEMPLATES[args.template]
    print(f"Using template: {args.template}")
elif args.prompt:
    prompt = args.prompt
else:
    # Default: show all templates
    print("\nNo prompt given. Generating from all AfriCode templates...\n")
    for name, template_prompt in PROMPT_TEMPLATES.items():
        print(f"\n{'='*55}")
        print(f"Template: {name.upper()}")
        print('='*55)
        result = generate_code(model, enc, template_prompt, device,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature,
                               top_k=args.top_k)
        print(result)
    return

# Generate
print(f"\nPrompt: {prompt}")
print("="*55)
for i in range(args.num_samples):
    if args.num_samples > 1:
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")
    result = generate_code(model, enc, prompt, device,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature,
                           top_k=args.top_k)
    print(result)
```

if **name** == “**main**”:
main()
