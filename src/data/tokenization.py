from datasets import load_dataset
from transformers import AutoTokenizer
import os
import os.path as op


def tokenization():
    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": op.join("data", "train.csv"),
            "validation": op.join("data", "val.csv"),
            "test": op.join("data", "test.csv"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return imdb_tokenized