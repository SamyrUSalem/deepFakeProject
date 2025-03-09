import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

FOLDER_CELEBA = "backend/datasets/celeba/img_align_celeba"
CSV_LABELS = "backend/datasets/celeba/list_attr_celeba.csv"

df = pd.read_csv(CSV_LABELS)
df = df[['image_id', 'Eyeglasses']]
df['Eyeglasses'] = df['Eyeglasses'].apply(lambda x: 1 if x == 1 else 0)

num_samples = min(df['Eyeglasses'].value_counts())
df_fake = df[df['Eyeglasses'] == 1]
df_real = df[df['Eyeglasses'] == 0].sample(num_samples, random_state=42)
df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)
df = df.sample(2000, random_state=42)

class CelebADataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(FOLDER_CELEBA, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

transformation = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

dataset_train = CelebADataset(df, transform=transformation)
loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)  

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_size = self._get_flatten_size()
        self.fc = nn.Linear(self.flatten_size, 1)

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.rand(1, 3, 128, 128)  
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
        print(f"🔍 New correct size for fc layer: {x.shape[1]}")  
        return x.shape[1] 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  

        print("Shape before fc layer:", x.shape) 

        x = self.fc(x)
        return x

def train_model():
    model = DeepFakeDetector()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(15): 
        print(f"Starting epoch {epoch+1}")

        for batch_idx, (images, labels) in enumerate(loader_train):
            optimizer.zero_grad()
            outputs = model(images).squeeze() 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0: 
                print(f"Epoch {epoch+1} - Batch {batch_idx} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "backend/deepfake_model.pth")
    print("Model trained and saved!")

train_model()
