'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
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
# TODO : Uncomment once our own gpt is implemented
# from models.gpt2 import GPT2Model

import torch.optim as optim
from transformers import GPT2Model as OpenAIGPT2Model
from transformers import GPT2Config
from transformers import GPT2Tokenizer

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt

from peft import get_peft_model, LoraConfig, TaskType

from visualization import plot_metrics, create_comparison_plots, plot_metrics_map

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

class ParaphraseGPT(nn.Module):
  """GPT-2 Model for paraphrase detection with PEFT support."""

  def __init__(self, args):
      super().__init__()
      # Load the base GPT-2 model
      
      self.gpt = OpenAIGPT2Model.from_pretrained(args.model_size)
      self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
      
      # Freeze all parameters in the model
      if args.peft_method != "full_finetune_classification_head" and args.peft_method != "full_finetune_embeddings":
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.classification_head = nn.Linear(args.d, 2)
        # Apply PEFT based on method choice
        if args.peft_method == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,      # Task type for PEFT
                inference_mode=False,
                r=args.lora_r,                  # Rank of the update matrices
                lora_alpha=args.lora_alpha,     # Alpha parameter for LoRA scaling
                lora_dropout=args.lora_dropout, # Dropout probability for LoRA layers
                target_modules=args.target_modules  # Modules to apply LoRA to
            )
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.peft_type = "lora"
        elif args.peft_method == "prefix_tuning":
            from peft import PrefixTuningConfig
            
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=args.num_virtual_tokens,
                #prefix_dropout=args.prefix_dropout,
                encoder_hidden_size=args.d
            )
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.peft_type = "prefix_tuning"
      else:
        # Unfreeze parameters if not using PEFT
        if args.peft_method == "full_finetune_classification_head":
          self.peft_type = "full_finetune_classification_head"
          self.classification_head = nn.Linear(args.d, 2)
        else:
          self.peft_type = "full_finetune_embeddings"
        for param in self.gpt.parameters():
            param.requires_grad = True
      

  def forward(self, input_ids, attention_mask, **kwargs):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    # Remove labels if they exist because the base model doesn't accept them.
    kwargs.pop("labels", None)

    if self.peft_type == "full_finetune_embeddings":
      output = self.gpt(input_ids=input_ids, attention_mask=attention_mask, **kwargs)  # [batch_size, seq_len, hidden_size]
      last_hidden_state = output['last_hidden_state']  

      # extract the hidden state of the token that is the first non-padding token in each sequence
      last_non_pad_idx = attention_mask.sum(dim=1) - 1
      last_token = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_non_pad_idx]

      # get the embedding matrix
      embedding_matrix = self.gpt.get_input_embeddings().weight

      # get the logits using the embedding matrix and no binary head
      logits = torch.matmul(last_token, embedding_matrix.T)
    else:
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
      logits = self.classification_head(last_token)

    return logits
  
  def get_trainable_parameters(self):
        """Count and print trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return {
            "trainable_params": trainable_params,
            "all_params": all_param,
            "trainable_percentage": 100 * trainable_params / all_param
        }

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

def train(args, return_metrics=False):
    """Train GPT-2 for paraphrase detection on the Quora dataset with PEFT support."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Create the data and its corresponding datasets and dataloader
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)
    
    # Use subset of data for quick experiments if specified
    if args.experiment_size > 0:
        para_train_data = para_train_data[:args.experiment_size]
        para_dev_data = para_dev_data[:args.experiment_size]
    
    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)
    
    # Get and log parameter counts
    param_stats = model.get_trainable_parameters()
    print(f"PEFT Method: {args.peft_method}")
    print(f"Total parameters: {param_stats['all_params']:,}")
    print(f"Trainable parameters: {param_stats['trainable_params']:,} ({param_stats['trainable_percentage']:.2f}%)")

    # Only optimize trainable parameters
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], 
                     lr=args.lr, weight_decay=0.01)

    total_steps = len(para_train_dataloader) * args.epochs

    # Linear scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    best_dev_acc = 0

    scaler = torch.amp.GradScaler("cuda") if args.use_gpu else None

    # Lists to record metrics over epochs
    epoch_losses = []
    epoch_accs = []
    epoch_f1s = []

    # Run for the specified number of epochs
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

            # Compute the loss, gradients, and update the model's parameters
            optimizer.zero_grad()

            if args.use_gpu:
                # Use autocast to reduce memory usage
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=True):
                    logits = model(b_ids, b_mask)
                    loss = F.cross_entropy(logits, labels, reduction='mean')

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(b_ids, b_mask)
                loss = F.cross_entropy(logits, labels, reduction='mean')
                
                loss.backward()
                optimizer.step()
                
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        epoch_losses.append(train_loss)
        epoch_accs.append(dev_acc)
        epoch_f1s.append(dev_f1)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
    
    # Generate epochs range for plotting
    epochs_range = range(1, args.epochs + 1)
    
    # Plot metrics
    #plot_metrics(epochs_range, epoch_losses, epoch_accs, epoch_f1s, args)
    
    if return_metrics:
        return model, optimizer, {
            "epoch_losses": epoch_losses,
            "epoch_accs": epoch_accs,
            "epoch_f1s": epoch_f1s,
            "final_train_loss": epoch_losses[-1],
            "final_dev_acc": epoch_accs[-1],
            "final_dev_f1": epoch_f1s[-1],
            "best_dev_acc": best_dev_acc,
            "trainable_params": param_stats["trainable_params"],
            "all_params": param_stats["all_params"],
            "trainable_percentage": param_stats["trainable_percentage"]
        }
    
    return model, optimizer

@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  
  para_dev_data = load_paraphrase_data(args.para_dev)[:args.experiment_size]
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  if args.experiment_size > 0:
    para_dev_data = para_dev_data[:args.experiment_size]
    para_dev_data = para_dev_data[:args.experiment_size]

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

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2-large')

  parser.add_argument("--experiment_size", type=int, default=1500,
                      help="Number of examples to use for quick experiments (set to -1 for full dataset)")
  

  # PEFT method selection
  parser.add_argument("--peft_method", type=str, default="full_finetune_embeddings",
                    choices=['lora', 'prefix_tuning', 'full_finetune_classification_head', 'full_finetune_embeddings'],
                    help="Parameter-efficient fine-tuning method")
  
  # LoRA specific arguments
  parser.add_argument("--lora_r", type=int, default=8, 
                    help="Rank dimension for LoRA")
  parser.add_argument("--lora_alpha", type=int, default=32, 
                    help="Alpha parameter for LoRA")
  parser.add_argument("--lora_dropout", type=float, default=0.1, 
                    help="Dropout for LoRA layers")
  
  # Adapter specific arguments
  parser.add_argument("--adapter_r", type=int, default=8, 
                    help="Bottleneck dimension for Adapter")
  parser.add_argument("--adapter_dropout", type=float, default=0.1, 
                    help="Dropout for Adapter layers")
  
  # Prefix Tuning specific arguments
  parser.add_argument("--num_virtual_tokens", type=int, default=20, 
                    help="Number of virtual tokens for prefix tuning")
  parser.add_argument("--prefix_dropout", type=float, default=0.1, 
                    help="Dropout for prefix tuning")
  
  # Which layers to target with PEFT
  parser.add_argument("--target_modules", type=str, nargs='+', 
                    default=["c_attn", "c_proj"],
                    help="List of module names to apply PEFT to")
  
  # Compare methods automatically
  parser.add_argument("--compare_methods", action='store_true',
                    help="Run experiments with all PEFT methods for comparison")
  
  # Compare lora methods automatically
  parser.add_argument("--compare_lora", action='store_true',
                    help="Run lora experiments with different parameters for comparison")
  
  args = parser.parse_args()

  return args

def compare_peft_methods(args):
    """Run experiments with different PEFT methods and compare results."""
    methods = ["lora", "prefix_tuning", "full_finetune_classification_head", "full_finetune_embeddings"]
    results = {}
    
    print("=" * 80)
    print("STARTING PEFT METHODS COMPARISON")
    print("=" * 80)
    
    # Store original arguments
    original_method = args.peft_method
    original_filepath = args.filepath
    
    # Create small subset of data for quick experiments
    if args.experiment_size > 0:
        print(f"Using {args.experiment_size} examples for experiments")
    
    for method in methods:
        print(f"\n{'-' * 30}")
        print(f"TESTING METHOD: {method}")
        print(f"{'-' * 30}")
        
        # Update args for this method
        args.peft_method = method
        if args.peft_method != "full_finetune_classification_head" and args.peft_method != "full_finetune_embeddings":
          args.d = 1280
          args.lr = 1e-4
        args.filepath = f'{args.epochs}-{args.lr}-{method}-paraphrase.pt'  # Save path
        
        # Train and evaluate
        model, optimizer, metrics = train(args, return_metrics=True)
        
        # Test and get results
        test_metrics = test(args)
        
        # Store results
        results[method] = {
            "train_loss": metrics["final_train_loss"],
            "dev_acc": metrics["final_dev_acc"],
            "dev_f1": metrics["final_dev_f1"],
            "test_acc": test_metrics["test_acc"] if test_metrics else None,
            "trainable_params": metrics["trainable_params"],
            "all_params": metrics["all_params"],
            "trainable_percentage": metrics["trainable_percentage"],
            "lr": args.lr
        }
    
    # Restore original arguments
    args.peft_method = original_method
    args.filepath = original_filepath
    
    # Print comparison
    print("\n" + "=" * 80)
    print("PEFT METHODS COMPARISON RESULTS")
    print("=" * 80)
    
    # Create a pretty table
    from tabulate import tabulate
    headers = ["Method", "Train Loss", "Dev Acc", "Dev F1", "Trainable %", "Trainable Params", "All Params"]
    table = []
    
    for method, metric in results.items():
        row = [
            method,
            f"{metric['train_loss']:.4f}",
            f"{metric['dev_acc']:.4f}",
            f"{metric['dev_f1']:.4f}",
            f"{metric['trainable_percentage']:.2f}%",
            f"{metric['trainable_params']:,}",
            f"{metric['all_params']:,}",
            f"{metric['lr']:,}",

        ]
        table.append(row)
    
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Create plots for visualization
    create_comparison_plots(results, args)
    
    return results

def compare_lora_methods(lora_configs, args, title):
    """Run experiments with different LoRA hyperparameter configurations and compare results."""
    # Define a list of LoRA hyperparameter configurations to test
    
    results = {}
    
    print("=" * 80)
    print("STARTING LoRA HYPERPARAMETER COMPARISON")
    print("=" * 80)
    
    # Store original hyperparameter settings
    original_lora_r = args.lora_r
    original_lora_alpha = args.lora_alpha
    original_lr = args.lr
    original_filepath = args.filepath
    
    # Set the PEFT method to "lora"
    args.peft_method = "lora"
    
    if args.experiment_size > 0:
        print(f"Using {args.experiment_size} examples for experiments")
    
    metrics_map = {}
    # Iterate over each LoRA configuration
    for config in lora_configs:
        args.lora_r = config["lora_r"]
        args.lora_alpha = config["lora_alpha"]
        args.lr = config["lr"]
        if title == "learning_rate":
          title_arg = f"{config['lr']}"
        elif title == "lora_r":
          title_arg = f"{config['lora_r']}"
        elif title == "lora_alpha":
          title_arg = f"{config['lora_alpha']}"
        # Update filepath to include hyperparameter info for clarity
        args.filepath = f'{title}_{args.epochs}-{args.lr}-lora_r{args.lora_r}_alpha{args.lora_alpha}-paraphrase.pt'
        
        print(f"\n{'-' * 30}")
        print(f"TESTING {title} LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}, lr={args.lr}")
        print(f"{'-' * 30}")
        
        # Train and evaluate the model using the current LoRA settings
        model, optimizer, metrics = train(args, return_metrics=True)
        # Optionally, run test() to generate predictions (side-effects such as saving files)
        #test(args)
        
        # Store the results
        config_name = f"lora_{title}_{title_arg}"
        metrics_map[config_name] = metrics
        results[config_name] = {
            "train_loss": metrics["final_train_loss"],
            "dev_acc": metrics["final_dev_acc"],
            "dev_f1": metrics["final_dev_f1"],
            "trainable_params": metrics["trainable_params"],
            "all_params": metrics["all_params"],
            "trainable_percentage": metrics["trainable_percentage"],
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout
        }
    
    # Restore original hyperparameter settings
    args.lora_r = original_lora_r
    args.lora_alpha = original_lora_alpha
    args.lr = original_lr
    args.filepath = original_filepath
    
    # Print comparison results
    print("\n" + "=" * 80)
    print("LoRA HYPERPARAMETER COMPARISON RESULTS")
    print("=" * 80)

    plot_metrics_map(metrics_map, f"lora_{title}")
    
    from tabulate import tabulate
    headers = ["Config", "Train Loss", "Dev Acc", "Dev F1", "Trainable %", "All Params", "lr", "lora_r", "lora_alpha"]
    table = []
    for config_name, metric in results.items():
        row = [
            config_name,
            f"{metric['train_loss']:.4f}",
            f"{metric['dev_acc']:.4f}",
            f"{metric['dev_f1']:.4f}",
            f"{metric['trainable_percentage']:.2f}%",
            f"{metric['all_params']:,}",
            f"{metric['lr']}",
            f"{metric['lora_r']}",
            f"{metric['lora_alpha']}",
        ]
        table.append(row)
    
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    return results

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
    print("Running the paraphrase detection script with PEFT support.")
    args = get_args()
    print(args)
    
    # Set filepath based on PEFT method
    args.filepath = f'{args.epochs}-{args.lr}-{args.peft_method}-paraphrase.pt'
    
    # Fix the seed for reproducibility
    seed_everything(args.seed)
    
    # If comparison mode is enabled, run comparison experiments
    if args.compare_methods:
        results = compare_peft_methods(args)
    elif args.compare_lora:
        lora_configs = [
        {"lora_r": 8, "lora_alpha": 32, "lr": 1e-5},
        {"lora_r": 8, "lora_alpha": 32, "lr": 1e-4},
        {"lora_r": 8, "lora_alpha": 32, "lr": 1e-3},
        ]
        results = compare_lora_methods(lora_configs, args, "learning_rate")
        lora_configs = [
        {"lora_r": 8, "lora_alpha": 32, "lr": 1e-4},
        {"lora_r": 16, "lora_alpha": 32, "lr": 1e-4},
        {"lora_r": 32, "lora_alpha": 32, "lr": 1e-4},
        ]
        results = compare_lora_methods(lora_configs, args, "lora_r")
        lora_configs = [
        {"lora_r": 32, "lora_alpha": 16, "lr": 1e-4},
        {"lora_r": 32, "lora_alpha": 32, "lr": 1e-4},
        {"lora_r": 32, "lora_alpha": 64, "lr": 1e-4},
        ]
        results = compare_lora_methods(lora_configs, args, "lora_alpha")
    else:
        # Run single method training
        model, optimizer = train(args)
        test(args)
        
    print("Done!")
