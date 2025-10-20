import torch
import os
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    """
    A custom Dataset to load pre-generated latent samples.
    It infers the label file path from the data file path
    by replacing 'train_data' with 'train_labels'.
    """
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the .pt file with data 
                                (e.g., '.../latent_train_data_TIMESTAMP.pt')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.data_path = data_path
        
        # --- Infer label path from data path ---
        try:
            # Replaces the specific 'train_data' part of the filename
            self.label_path = self.data_path.replace("train_data", "train_labels")
            
            # Sanity check 1: Ensure the replacement actually changed the string
            if self.label_path == self.data_path:
                raise ValueError("Filename convention error: 'train_data' not found in path.")
            
            # Sanity check 2: Check if the inferred file actually exists
            if not os.path.exists(self.label_path):
                 raise FileNotFoundError(f"Inferred label file not found at: {self.label_path}")
                 
        except Exception as e:
            print(f"Error inferring label path from data path: {self.data_path}")
            print(f"Error details: {e}")
            raise
        
        # --- Load data and targets ---
        try:
            self.data = torch.load(self.data_path)
            self.targets = torch.load(self.label_path)
        except FileNotFoundError as e:
            print(f"Error loading files.")
            print(f"Data path: {self.data_path}")
            print(f"Inferred Label path: {self.label_path}")
            raise
            
        print(f"Successfully loaded data from {self.data_path}")
        print(f"Successfully inferred and loaded labels from {self.label_path}")
        print(f"Data shape: {self.data.shape}, Labels shape: {self.targets.shape}")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets one sample.
        Returns a tuple (data, target).
        """
        sample = self.data[idx]
        target = self.targets[idx]
        
        # Apply the transform to the data (the latent sample)
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target