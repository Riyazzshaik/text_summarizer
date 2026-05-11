import torch
from tqdm import tqdm
from datasets import load_from_disk
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def generate_batch_sized_chunks(self, list_of_elements, batch_size):

        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]


    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=2,
        device="cpu",
        column_text="dialogue",
        column_summary="summary"
    ):

        article_batches = list(
            self.generate_batch_sized_chunks(
                dataset[column_text],
                batch_size
            )
        )

        target_batches = list(
            self.generate_batch_sized_chunks(
                dataset[column_summary],
                batch_size
            )
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches)
        ):

            inputs = tokenizer(
                article_batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=4,
                max_length=64
            )

            decoded_summaries = [
                tokenizer.decode(
                    s,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                for s in summaries
            ]

            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch
            )

        score = metric.compute()

        return score


    def evaluate(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)

        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )

        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset=dataset_samsum_pt["test"].select(range(20)),
            metric=rouge_metric,
            model=model,
            tokenizer=tokenizer,
            batch_size=1,
            device=device
        )

        print("\nROUGE SCORE:\n")
        print(score)