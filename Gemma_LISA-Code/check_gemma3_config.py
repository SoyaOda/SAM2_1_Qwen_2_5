#!/usr/bin/env python3
"""
Check Gemma3Config structure and attributes
"""

from transformers import AutoConfig, AutoTokenizer

print("🔍 Checking Gemma3Config Structure")
print("="*50)

# Load config
config = AutoConfig.from_pretrained("google/gemma-3-4b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

print(f"Config Type: {type(config)}")
print(f"Config Class: {config.__class__.__name__}")

print("\n📋 All Config Attributes:")
attrs = [attr for attr in dir(config) if not attr.startswith('_')]
for attr in sorted(attrs):
    try:
        value = getattr(config, attr, "N/A")
        if not callable(value):
            print(f"   {attr}: {value}")
    except:
        print(f"   {attr}: <error accessing>")

print("\n🔑 Key Attributes Check:")
key_attrs = [
    'model_type', 'vocab_size', 'pad_token_id', 'hidden_size', 
    'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
    'max_position_embeddings', 'rms_norm_eps'
]

for attr in key_attrs:
    value = getattr(config, attr, "❌ NOT FOUND")
    print(f"   {attr}: {value}")

print(f"\n🔤 Tokenizer Info:")
print(f"   Vocab Size: {tokenizer.vocab_size}")
print(f"   Pad Token: {tokenizer.pad_token}")
print(f"   EOS Token: {tokenizer.eos_token}")

print(f"\n💡 Vocab Size Source:")
print(f"   From tokenizer: {tokenizer.vocab_size}")
print(f"   From config: {getattr(config, 'vocab_size', 'N/A')}")

# Check if there are alternative attribute names
print(f"\n🔍 Alternative Vocab Size Attributes:")
alt_attrs = ['vocabulary_size', 'n_vocab', 'vocab_length']
for attr in alt_attrs:
    value = getattr(config, attr, "Not found")
    print(f"   {attr}: {value}") 