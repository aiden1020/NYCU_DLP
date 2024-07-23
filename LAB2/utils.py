# script for drawing figures, and more if needed
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_metrics(args):

    if args.mode == 'SD':
        csv_path = 'result/SD_result.csv'
        method = 'Subject Dependent'
        fig_path = 'graph/SD_graph.png'
    if args.mode == 'LOSO':
        csv_path = 'result/LOSO_result.csv'
        method = 'LOSO'
        fig_path = 'graph/LOSO_graph.png'
    if args.mode == 'FT':
        csv_path = 'result/FT_result.csv'
        method = 'LOSO with Finetuning'
        fig_path = 'graph/FT_graph.png'
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{method} Train Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', color='green')
    plt.plot(df['epoch'], df['test_accuracy'], label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{method} Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script for drawing figures')
    parser.add_argument('--mode', type=str, required=True, choices=['SD', 'LOSO', 'FT'],
                        help="Choose training mode: 'SD' for Subject Dependent, 'LOSO' for Leave-One-Subject-Out, 'FT' for Fine-tune")
    args = parser.parse_args()
    plot_metrics(args)