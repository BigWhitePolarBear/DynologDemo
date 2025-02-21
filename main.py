import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from data import prepare_dataset, format_dataset, custom_collate
from utils import setup_distributed_env, cleanup_distributed_env

def main():
    device, rank, _ = setup_distributed_env()

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
                              sampler=train_sampler, collate_fn=custom_collate, 
                              num_workers=4, pin_memory=True)

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
        model.module.save_pretrained("finetuned-llama-3.1-8b")
        tokenizer.save_pretrained("finetuned-llama-3.1-8b")

    cleanup_distributed_env()

if __name__ == "__main__":
    main()