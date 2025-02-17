import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from data import prepare_dataset, format_dataset, custom_collate

# using Hugging Face datasets to load data
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
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

# get model num layers
num_layers = model.config.num_hidden_layers

# freeze some layers
half_layers = int(num_layers * 0.8)
for i in range(half_layers):
    for param in model.model.layers[i].parameters():
        param.requires_grad = False

# define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*3)

# loss function
criterion = nn.CrossEntropyLoss()

num_epochs = 1
# finetune loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
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
        state = optimizer.state_dict()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# saving the finetuned model
model.save_pretrained("finetuned-llama-3.1-8b")
tokenizer.save_pretrained("finetuned-llama-3.1-8b")