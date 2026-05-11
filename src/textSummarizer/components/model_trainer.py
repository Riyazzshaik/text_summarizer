from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch
import os


class ModelTrainer:

    def __init__(self, config):
        self.config = config

    def train(self):

        # device setup
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "t5-small"
        )

        # model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5-small"
        ).to(device)

        # data collator
        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model
        )

        # load transformed dataset
        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )

        print(dataset_samsum_pt)

        # VERY SMALL DATASET FOR FAST TRAINING
        train_data = dataset_samsum_pt["train"].select(range(20))
        eval_data = dataset_samsum_pt["validation"].select(range(5))

        # training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,

            num_train_epochs=1,

            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,

            logging_steps=1,

            save_steps=5,

            max_steps=10,

            report_to="none"
        )

        # trainer
        trainer = Trainer(
            model=model,
            args=trainer_args,

            train_dataset=train_data,
            eval_dataset=eval_data,

            data_collator=seq2seq_data_collator
        )

        # start training
        trainer.train()

        # save model
        model.save_pretrained(
            os.path.join(
                self.config.root_dir,
                "t5-small-model"
            ),
            safe_serialization=False
        )

        # save tokenizer
        tokenizer.save_pretrained(
            os.path.join(
                self.config.root_dir,
                "tokenizer"
            )
        )

        print("Model training completed successfully")