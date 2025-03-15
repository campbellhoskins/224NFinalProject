import matplotlib.pyplot as plt


def create_comparison_plots(results, args):
    """Create plots to visualize comparison results."""
    # Get methods and metrics
    methods = list(results.keys())
    
    # Create accuracy comparison plot
    plt.figure(figsize=(10, 6))
    
    # Add bars for dev accuracy
    plt.bar(
        [i - 0.2 for i in range(len(methods))], 
        [results[m]["dev_acc"] for m in methods],
        width=0.4,
        label="Dev Accuracy"
    )
    
    # Add bars for dev F1
    plt.bar(
        [i + 0.2 for i in range(len(methods))], 
        [results[m]["dev_f1"] for m in methods],
        width=0.4,
        label="Dev F1"
    )
    
    plt.xlabel("PEFT Method")
    plt.ylabel("Score")
    plt.title("Performance Comparison of PEFT Methods")
    plt.xticks(range(len(methods)), methods)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("peft_performance_comparison.png")
    
    # Create parameter efficiency plot (log scale)
    plt.figure(figsize=(10, 6))
    
    # Calculate trainable parameters percentage
    trainable_percentages = [results[m]["trainable_percentage"] for m in methods]
    
    plt.bar(methods, trainable_percentages)
    plt.xlabel("PEFT Method")
    plt.ylabel("Trainable Parameters (%)")
    plt.title("Parameter Efficiency of PEFT Methods")
    plt.yscale("log")  # Log scale to better visualize the differences
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("peft_parameter_efficiency.png")
    
    print(f"Comparison plots saved to: peft_performance_comparison.png and peft_parameter_efficiency.png")

def plot_metrics(epochs, losses, accs, f1s, args):
    """Plot training and evaluation metrics."""
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot training loss
    ax1.plot(epochs, losses, marker='o', color='red', label=f'{args.peft_method}')
    ax1.set_title(f'Training Loss per Epoch ({args.peft_method})')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot dev accuracy
    ax2.plot(epochs, accs, marker='o', color='blue', label=f'{args.peft_method}')
    ax2.set_title(f'Validation Accuracy per Epoch ({args.peft_method})')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    # Plot dev F1 score
    ax3.plot(epochs, f1s, marker='o', color='green', label=f'{args.peft_method}')
    ax3.set_title(f'Validation F1 Score per Epoch ({args.peft_method})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{args.peft_method}_lr{args.lr}_r{args.lora_r}_alpha{args.lora_alpha}metrics.png')
    plt.close()

def plot_metrics_map(metrics_map, title):
    """
    Plot training and evaluation metrics for multiple training sessions on the same figure.

    Parameters:
    -----------
    metrics_map : dict
        A dictionary where each key is a configuration name (e.g. a string representing the model or method)
        and each value is a dictionary with keys:
            - 'epochs': list or range of epoch numbers (optional, if missing it will default to range based on losses length)
            - 'losses': list of training losses per epoch
            - 'accs': list of validation accuracies per epoch
            - 'f1s': list of validation F1 scores per epoch
    """
    import matplotlib.pyplot as plt

    # Create figure with 3 subplots, one for each metric
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Iterate over each configuration and plot its metrics
    for config, metrics in metrics_map.items():
        # Use the provided epochs or create a default range based on the length of losses
        epochs = metrics.get('epochs', list(range(1, len(metrics['epoch_losses']) + 1)))
        losses = metrics['epoch_losses']
        accs = metrics['epoch_accs']
        f1s = metrics['epoch_f1s']
        
        ax1.plot(epochs, losses, marker='o', label=config)
        ax2.plot(epochs, accs, marker='o', label=config)
        ax3.plot(epochs, f1s, marker='o', label=config)

    # Formatting for training loss plot
    ax1.set_title('Training Loss per Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Formatting for validation accuracy plot
    ax2.set_title('Validation Accuracy per Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    # Formatting for validation F1 score plot
    ax3.set_title('Validation F1 Score per Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'{title}_comparison.png')
    plt.close()
