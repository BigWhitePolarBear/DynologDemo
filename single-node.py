import time
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from data import prepare_dataset, format_dataset, custom_collate, truncate_dataset

def main(args):
    # using Hugging Face datasets to load data
    dataset = load_dataset(args.dataset_id)
    truncate_dataset(dataset, args.data_ratio)
    dataset = dataset.map(format_dataset)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loding model and tokenizer
    model = LlamaForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # define dataset and loader
    train_dataset = prepare_dataset(dataset["train"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate)

    # freeze some layers
    num_layers = model.config.num_hidden_layers
    freeze_layers = int(num_layers * args.freeze_layers_ratio)
    for i in range(freeze_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = False

    # define optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps, 
                                                num_training_steps=len(train_loader)*args.num_epochs)

    # finetune loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", 
                          bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]'):
            # moving input to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * args.batch_size
            
            # backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # saving the finetuned model
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model using PyTorch DDP")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B", help="Model ID from Hugging Face")
    parser.add_argument("--dataset_id", type=str, default="databricks/databricks-dolly-15k", help="Dataset ID to train on")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--freeze_layers_ratio", type=float, default=0.5, help="Ratio of layers to freeze (0.0 to 1.0)")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--output_dir", type=str, default="finetuned-model", help="Output directory to save the model")
    parser.add_argument("--data_ratio", type=float, default=1.0, help="Ratio of data to use for training")

    args = parser.parse_args()

    main(args)