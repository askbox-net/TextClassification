from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from transformers import set_seed, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


checkpoint = "HuggingFaceTB/SmolLM-135M"


if __name__ == '__main__':
    imdb = load_dataset("imdb")

    model_config = AutoConfig.from_pretrained(checkpoint, num_labels=2) # Binary Classification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=model_config)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side = "left" # Very Important
    tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    print(imdb)

    print(imdb.keys())
    """
    training_args = TrainingArguments(
        output_dir="smollm-135m",
        learning_rate=2e-5,
        per_device_train_batch_size=1, #16,
        per_device_eval_batch_size=1, #16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    """

    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

    classifier = pipeline("sentiment-analysis", model="smollm-135m/checkpoint-50000")
    print(classifier(text))

