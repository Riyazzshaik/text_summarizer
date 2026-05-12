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
        BASE_DIR = os.getcwd()

        # TOKENIZER PATH
        self.tokenizer_path = os.path.join(
            BASE_DIR,
            "artifacts",
            "model_trainer",
            "tokenizer"
        )

        # MODEL PATH
        self.model_path = os.path.join(
            BASE_DIR,
            "artifacts",
            "model_trainer",
            "t5-small-model"
        )

        print(f"\nLoading tokenizer from: {self.tokenizer_path}")
        print(f"Loading model from: {self.model_path}")

        # LOAD TOKENIZER
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=True
        )

        # LOAD MODEL
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            local_files_only=True
        ).to(self.device)

        print(f"Model loaded successfully on {self.device}\n")

    def predict(self, text: str):

        # ADD PREFIX FOR T5
        input_text = "summarize: " + text

        # TOKENIZE INPUT
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # MOVE TO DEVICE
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # GENERATE SUMMARY
        summary_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=35,
            min_length=8,
            num_beams=8,
            length_penalty=2.0,
            early_stopping=True
        )
        # DECODE SUMMARY
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return summary