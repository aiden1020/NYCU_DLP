# implement your training script here
import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SCCNet import SCCNet 
from Dataloader import MIBCI2aDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main(args):
    if args.mode == 'SD':
        train_dataset = MIBCI2aDataset(mode='SD_train')
        test_dataset = MIBCI2aDataset(mode='SD_test')
    elif args.mode == 'LOSO':
        train_dataset = MIBCI2aDataset(mode='LOSO_train')
        test_dataset = MIBCI2aDataset(mode='LOSO_test')
    elif args.mode == 'FT':
        train_dataset = MIBCI2aDataset(mode='finetune')
        test_dataset = MIBCI2aDataset(mode='LOSO_test')
    else:
        raise ValueError("Invalid mode. Choose from ['SD', 'LOSO', 'FT']")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SCCNet(timeSample=train_dataset.features.shape[2],C=train_dataset.features.shape[1],Nc=args.Nc, Nt=args.Nt, Nu=args.Nu, dropoutRate=args.dropoutRate).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

    num_epochs = args.num_epochs
    best_accuracy = 0.0
    best_model_path = "best_SCCNet_model.pth"
    
    with open(args.csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'test_accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
            test_accuracy = evaluate_model(model, test_loader, device)
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model, best_model_path)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            writer.writerow({'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy})

    print(f"Best model saved with accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SCCNet with different modes.')
    parser.add_argument('--mode', type=str, required=True, choices=['SD', 'LOSO', 'FT'],
                        help="Choose training mode: 'SD' for Subject Dependent, 'LOSO' for Leave-One-Subject-Out, 'FT' for Fine-tune")
    parser.add_argument('--Nc', type=int, default=22, help="Number of channels")
    parser.add_argument('--Nt', type=int, default=438, help="Number of time points")
    parser.add_argument('--Nu', type=int, default=16, help="Number of convolutional kernels")
    parser.add_argument('--dropoutRate', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to save CSV file")

    args = parser.parse_args()
    main(args)
