#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer on the provided dataset.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm

def train_tokenizer(
    input_files: List[str],
    output_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
    user_defined_symbols: Optional[List[str]] = None
):
    """
    Train a SentencePiece tokenizer.

    Args:
        input_files: List of paths to input text files.
        output_prefix: Prefix for the output model and vocab files.
        vocab_size: Vocabulary size.
        model_type: Model type (unigram, bpe, char, word).
        character_coverage: Character coverage amount.
        pad_id: Padding token ID.
        unk_id: Unknown token ID.
        bos_id: Beginning of sentence token ID.
        eos_id: End of sentence token ID.
        user_defined_symbols: List of user defined symbols.
    """
    # Ensure input files exist
    valid_inputs = []
    for f in input_files:
        if os.path.exists(f):
            valid_inputs.append(f)
        else:
            print(f"Warning: Input file not found: {f}")
    
    if not valid_inputs:
        print("Error: No valid input files found.")
        sys.exit(1)

    # Convert paths to comma-separated string
    input_argument = ",".join(valid_inputs)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f"--input={input_argument} "
        f"--model_prefix={output_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage={character_coverage} "
        f"--pad_id={pad_id} "
        f"--unk_id={unk_id} "
        f"--bos_id={bos_id} "
        f"--eos_id={eos_id} "
        f"--hard_vocab_limit=false"
    )

    if user_defined_symbols:
        cmd += f" --user_defined_symbols={','.join(user_defined_symbols)}"

    print(f"Training tokenizer with command: {cmd}")
    
    # Train the tokenizer
    try:
        spm.SentencePieceTrainer.train(cmd)
        print(f"Tokenizer trained successfully. Model saved to {output_prefix}.model")
    except Exception as e:
        print(f"Error training tokenizer: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Input files")
    parser.add_argument("--output", required=True, help="Output directory or prefix")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--model-type", default="bpe", choices=["unigram", "bpe", "char", "word"], help="Model type")
    
    args = parser.parse_args()

    # If output is a directory, append a default filename
    if os.path.isdir(args.output) or args.output.endswith("/"):
        output_prefix = os.path.join(args.output, "tokenizer")
    else:
        output_prefix = args.output

    train_tokenizer(
        input_files=args.input,
        output_prefix=output_prefix,
        vocab_size=args.vocab_size
    )

if __name__ == "__main__":
    main()
