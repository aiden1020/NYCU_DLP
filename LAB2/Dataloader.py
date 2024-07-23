import torch
import numpy as np
import os 
class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        all_files = [os.path.join(filePath, f) for f in os.listdir(filePath)]
        data = [np.load(f) for f in all_files]
        features = np.concatenate(data, axis=0)
        return features

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        all_files = [os.path.join(filePath, f) for f in os.listdir(filePath)]
        data = [np.load(f) for f in all_files]
        labels = np.concatenate(data, axis=0)
        return labels

    def __init__(self, mode):
        assert mode in ['SD_train', 'LOSO_train','finetune','SD_test','LOSO_test']
        if mode == 'SD_train':
            self.features = self._getFeatures(filePath='./dataset/SD_train/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_train/labels/')

        if mode == 'LOSO_train':
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')

        if mode == 'finetune':
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')

        if mode == 'SD_test':
            self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')

        if mode == 'LOSO_test':
            self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')

    def __len__(self):
        # implement the len method
        return len(self.features)


    def __getitem__(self, idx):
        # implement the getitem method
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":

    train_dataset = MIBCI2aDataset(mode='SD_train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


    test_dataset = MIBCI2aDataset(mode='SD_test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


    print(f"訓練數據集大小: {len(train_dataset)}")
    print(f"測試數據集大小: {len(test_dataset)}")


    for features, labels in train_loader:
        print(f"特徵形狀: {features.shape}")
        print(f"標籤形狀: {labels.shape}")
        break