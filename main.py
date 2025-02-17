import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from data import prepare_dataset, format_dataset, custom_collate

def setup():
    # initialize distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        num_gpus = int(os.environ['WORLD_SIZE'])  
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        num_gpus = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.distributed.init_process_group(backend='nccl', 
                                        init_method='env://',
                                        world_size=num_gpus,
                                        rank=rank)
    return device, rank, num_gpus

def cleanup():
    torch.distributed.destroy_process_group()

def main():
    device, rank, num_gpus = setup()

    # load dataset
    if rank == 0:
        dataset = load_dataset("databricks/databricks-dolly-15k")
    torch.distributed.barrier()
    if rank != 0:
        dataset = load_dataset("databricks/databricks-dolly-15k")
    dataset = dataset.map(format_dataset)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loding model and tokenizer
    model_id = "meta-llama/Llama-3.1-8B"
    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # define dataset and loader
    train_dataset = prepare_dataset(dataset["train"], tokenizer)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, # DistributedSampler handles shuffle
                              sampler=train_sampler, collate_fn=custom_collate, pin_memory=True)

    # freeze some layers
    num_layers = model.config.num_hidden_layers
    half_layers = int(num_layers * 0.9)
    for i in range(half_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = False

    # define DDP model
    model = DDP(model, device_ids=[device])

    # define optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    # learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*3)

    num_epochs = 1
    # finetune loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # set epoch for DistributedSampler to shuffle
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=(rank != 0)):
            # moving input to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # forward
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    if rank == 0:
        # saving the finetuned model
        model.save_pretrained("finetuned-llama-3.1-8b")
        tokenizer.save_pretrained("finetuned-llama-3.1-8b")

    cleanup()

if __name__ == "__main__":
    main()