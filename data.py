import torch

def truncate_dataset(dataset, data_ratio):
    if data_ratio < 1.0:
        dataset["train"] = dataset["train"].select(range(int(len(dataset["train"])*data_ratio)))

# formatting data
def format_dataset(example):
    return {
        "input": example["instruction"],
        "output": example["response"]
    }

def prepare_dataset(dataset, tokenizer):
    def process_data(example):
        # combining input and output
        text = f"{example['input']}{example['output']}"
        # tokenizing
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                           max_length=512, padding="max_length")
        # preparing targets
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    
    # preprocessing 
    return dataset.map(process_data, batched=False, remove_columns=["input", "output"])

def custom_collate(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]).squeeze(0) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]).squeeze(0) for item in batch])
    labels = torch.stack([torch.tensor(item["labels"]).squeeze(0) for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }