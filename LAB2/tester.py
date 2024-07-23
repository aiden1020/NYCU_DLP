# implement your testing script here
import torch
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset
from trainer import evaluate_model

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    return model

def test_model(mode, model_path, device):
    model = load_model(model_path, device)
    print(f"Model for {mode} loaded successfully.")
    model.eval()

    if mode == 'SD':
        test_dataset = MIBCI2aDataset(mode='SD_test')
    elif mode == 'LOSO':
        test_dataset = MIBCI2aDataset(mode='LOSO_test')
    elif mode == 'FT':
        test_dataset = MIBCI2aDataset(mode='LOSO_test')  
    else:
        raise ValueError("Invalid mode. Choose from ['SD', 'LOSO', 'FT']")
    
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy for {mode}: {test_accuracy:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = ['SD', 'LOSO', 'FT']
    model_paths = {
        'SD': 'best/mode_SD.pth',
        'LOSO': 'best/mode_LOSO.pth',
        'FT': 'best/mode_FT.pth'
    }
    
    for mode in modes:
        model_path = model_paths[mode]
        test_model(mode, model_path, device)
