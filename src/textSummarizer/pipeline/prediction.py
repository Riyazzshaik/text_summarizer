import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class PredictionPipeline:

    def __init__(self):

        # DEVICE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # BASE DIRECTORY
        BASE_DIR = os.path.abspath(os.getcwd())

        # TOKENIZER PATH
        tokenizer_path = os.path.join(
            BASE_DIR,
            "artifacts",
            "model_trainer",
            "tokenizer"
        )

        # MODEL PATH
        model_path = os.path.join(
            BASE_DIR,
            "artifacts",
            "model_trainer",
            "t5-small-model"
        )

        print(f"Loading tokenizer from: {tokenizer_path}")
        print(f"Loading model from: {model_path}")

        # LOAD TOKENIZER
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True
        )

        # LOAD MODEL
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)

    def predict(self, text):

        # TOKENIZE INPUT
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # GENERATE SUMMARY
        summary_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_length=50,
            min_length=10,
            num_beams=4,
            early_stopping=True
        )

        # DECODE SUMMARY
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary