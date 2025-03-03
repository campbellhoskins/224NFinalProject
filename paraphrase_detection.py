'''
Paraphrase detection for GPT starter code with LoRA PEFT implementation.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model with LoRA.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu --use_lora`
trains and evaluates your ParaphraseGPT model with LoRA PEFT and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase

import torch.optim as optim
from transformers import GPT2Model as OpenAIGPT2Model
from transformers import GPT2Config
from transformers import GPT2Tokenizer

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Import for LoRA implementation
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
except ImportError:
    print("PEFT library not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
except ImportError:
    print("PEFT library not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
import math

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# We'll implement a custom LoRA adaptation directly rather than using a separate class
# This is a reference implementation to understand how LoRA works
# def apply_lora_to_linear_layer(linear_layer, rank=8, alpha=16, dropout=0.1):
#     """
#     Apply LoRA to a nn.Linear layer directly without using the PEFT library
#     """
#     in_dim, out_dim = linear_layer.weight.shape
#     
#     # Create LoRA matrices
#     lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
#     lora_B = nn.Parameter(torch.zeros(rank, out_dim))
#     
#     # Store original weights and freeze them
#     original_forward = linear_layer.forward
#     linear_layer.weight.requires_grad = False
#     
#     # Add LoRA parameters to the layer
#     linear_layer.lora_A = lora_A
#     linear_layer.lora_B = lora_B
#     linear_layer.scaling = alpha / rank
#     linear_layer.lora_dropout = nn.Dropout(p=dropout)
#     
#     # Override forward method
#     def lora_forward(x):
#         orig_output = original_forward(x)
#         lora_output = (linear_layer.lora_dropout(x) @ linear_layer.lora_A 
#                       @ linear_layer.lora_B) * linear_layer.scaling
#         return orig_output + lora_output
#     
#     linear_layer.forward = lora_forward
#     return linear_layer


class ParaphraseGPT(nn.Module):
  """GPT-2 Model designed for paraphrase detection with LoRA PEFT support."""

  def __init__(self, args):
    super().__init__()
    self.gpt = OpenAIGPT2Model.from_pretrained(args.model_size)
    self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
    
    # Set requires_grad to False for all parameters by default
    for param in self.gpt.parameters():
      param.requires_grad = False
      
    # Apply LoRA if enabled
    if args.use_lora:
      # Configure LoRA
      peft_config = LoraConfig(
          task_type=TaskType.CAUSAL_LM,
          inference_mode=False,
          r=args.lora_rank,  # rank of the update matrices
          lora_alpha=args.lora_alpha,  # scaling factor
          lora_dropout=args.lora_dropout,
          target_modules=args.target_modules  # apply LoRA to attention matrices
      )
      # Apply LoRA to the model
      self.gpt = get_peft_model(self.gpt, peft_config)
      # Print trainable parameters stats
      self.paraphrase_detection_head = nn.Linear(args.d, 2)
      total_params = sum(p.numel() for p in self.parameters())
      trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print(f"Total parameters: {total_params:,}")
      print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    else:
      # Traditional fine-tuning if not using LoRA
      for param in self.gpt.parameters():
        param.requires_grad = True
      total_params = sum(p.numel() for p in self.parameters())
      trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print(f"Total parameters: {total_params:,}")
      print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")


  def forward(self, input_ids, attention_mask):
    """
    Predict the label of the token using GPT-2's output embeddings.

    We structure the input as:
      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
    """
    # When using LoRA through PEFT, we need to be careful with kwargs
    if hasattr(self.gpt, 'base_model'):
        # This is a PEFT model
        output = self.gpt.base_model.forward(input_ids=input_ids, attention_mask=attention_mask)
    else:
        # Regular GPT model
        output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    
    last_hidden_state = output['last_hidden_state']  

    # Extract the hidden state of the last non-padding token in each sequence
    last_non_pad_idx = attention_mask.sum(dim=1) - 1
    last_token = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_non_pad_idx]

    # Get the embedding matrix - need to handle PEFT model specially
    #if hasattr(self.gpt, 'base_model'):
    #    # For PEFT model
    #    embedding_matrix = self.gpt.get_input_embeddings().weight
    #else:
    #    # For regular model
    #    embedding_matrix = self.gpt.get_input_embeddings().weight

    # Get logits by dot product with embedding matrix
    #logits = torch.matmul(last_token, embedding_matrix.T)
    logits = self.paraphrase_detection_head(last_token)

    return logits


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)[:100]
  para_dev_data = load_paraphrase_data(args.para_dev)[:100]

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr

  # Set up optimizer with weight decay
  if args.use_lora:
    # For LoRA, only optimize the LoRA parameters
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
  else:
    # Traditional full fine-tuning
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

  total_steps = len(para_train_dataloader) * args.epochs

  # Using a linear scheduler with warmup
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=int(0.1 * total_steps),
      num_training_steps=total_steps
  )
  best_dev_acc = 0

  scaler = GradScaler()

  # Lists to record metrics over epochs.
  epoch_losses = []
  epoch_f1s = []

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the device
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()

      # Use autocast to reduce memory usage
      with autocast():
        logits = model(b_ids, b_mask)
        preds = torch.argmax(logits, dim=1)
        #print("Preds : ", preds)
        loss = F.cross_entropy(logits, labels, reduction='mean')
      #print("Labels : ", labels)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      # After loss.backward() or scaler.scale(loss).backward(), unscale gradients if using GradScaler:
      scaler.unscale_(optimizer)

      # Check if gradients for the classification head are computed:
      if model.paraphrase_detection_head.weight.grad is not None:
          grad_norm = model.paraphrase_detection_head.weight.grad.norm().item()
          print(f"Grad norm for classification head: {grad_norm:.4f}")

      # Optionally, log the weight norm before and after the optimizer step:
      weight_norm_before = model.paraphrase_detection_head.weight.norm().item()
      print(f"Weight norm before update: {weight_norm_before:.4f}")

      # Perform optimizer step:
      scaler.step(optimizer)
      scaler.update()

      # After the update, check the new weight norm:
      weight_norm_after = model.paraphrase_detection_head.weight.norm().item()
      print(f"Weight norm after update: {weight_norm_after:.4f}")

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    epoch_losses.append(train_loss)
    epoch_f1s.append(dev_f1)
    epochs = range(1, epoch + 2)  # +2 because epoch is 0-indexed and we want to start from 1
  
    print(f"Epoch {epoch+1}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
    if args.use_lora:
      print(f"Using LoRA with rank {args.lora_rank}, alpha {args.lora_alpha}")
  
  # Plot final metrics
  plot_metrics(epochs, epoch_losses, epoch_f1s, args)


def plot_metrics(epochs, losses, f1s, args):
  plt.figure()
  plt.plot(epochs, losses, marker='o')
  title_prefix = "LoRA " if args.use_lora else ""
  plt.title(f'{title_prefix}Training Loss per Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig(f'{"lora_" if args.use_lora else ""}train_loss.png')
  plt.close()

  plt.figure()
  plt.plot(epochs, f1s, marker='o')
  plt.title(f'{title_prefix}Validation F1 Score per Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('F1 Score')
  plt.savefig(f'{"lora_" if args.use_lora else ""}dev_f1.png')
  plt.close()


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  try:
    saved = torch.load(args.filepath, map_location=device)
    
    # Recreate args with the saved parameters
    saved_args = saved['args']
    
    # Ensure use_lora is properly set for loading
    if not hasattr(saved_args, 'use_lora'):
      saved_args.use_lora = False
    
    # Create model and load state dict
    model = ParaphraseGPT(saved_args)
    
    # Handle potential incompatibility with PEFT models
    try:
      model.load_state_dict(saved['model'])
    except Exception as e:
      print(f"Error loading state dict directly: {e}")
      print("Trying alternative loading method...")
      
      # If we're using LoRA, we might need to handle loading differently
      if saved_args.use_lora:
        # Initialize a fresh model and manually copy weights
        for name, param in model.named_parameters():
          if name in saved['model']:
            param.data = saved['model'][name]
    
    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")
    
    para_dev_data = load_paraphrase_data(args.para_dev)[:100]
    para_test_data = load_paraphrase_data(args.para_test, split='test')
  
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)
  
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)
    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)
  
    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
    print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)
  
    with open(args.para_dev_out, "w+") as f:
      f.write(f"id \t Predicted_Is_Paraphrase \n")
      for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
        f.write(f"{p}, {s} \n")
  
    with open(args.para_test_out, "w+") as f:
      f.write(f"id \t Predicted_Is_Paraphrase \n")
      for p, s in zip(test_para_sent_ids, test_para_y_pred):
        f.write(f"{p}, {s} \n")
        
  except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  
  # LoRA specific arguments
  parser.add_argument("--use_lora", action='store_true', help="Whether to use LoRA PEFT")
  parser.add_argument("--lora_rank", type=int, default=8, help="Rank of LoRA update matrices")
  parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor for LoRA")
  parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
  parser.add_argument("--target_modules", nargs='+', default=["c_attn", "c_proj"], 
                      help="Target modules for LoRA (default: attention layers)")

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  print("Running the paraphrase detection script with LoRA PEFT support.")
  args = get_args()
  print(args)
  lora_tag = "_lora" if args.use_lora else ""
  args.filepath = f'{args.epochs}-{args.lr}{lora_tag}-paraphrase.pt'  # Save path with LoRA indicator
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)