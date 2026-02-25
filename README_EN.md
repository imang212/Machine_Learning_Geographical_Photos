# Training Model for Geographic Data
I decided to create a machine learning training model for recognizing
geographic data as part of a seminar project at USU.

## Dataset
I am working with a dataset from Kaggle (planets_dataset), which
contains approximately 40,000 test images and 40,000 training images of
Amazon rainforest terrain. It also includes CSV files describing both
the training and test images.

The dataset contains geographic images taken in the Amazon rainforest,
along with two CSV files that describe what appears in each image for
machine learning purposes.

The dataset can be freely used for machine learning under the Database
Contents License.

Dataset link:
https://www.kaggle.com/datasets/nikitarom/planets-dataset/data

The root structure of the dataset:

![image](https://github.com/user-attachments/assets/1fab1cc9-de39-4a88-a92f-6ae9f64b1def)

### CSV Files
**sample_submission.csv** - 61,191 records - 2 columns: - image_name --
name of the image - tags -- description of features present in the image

**train_classes.csv** - 40,479 records - 2 columns: - image_name -- name
of the image - tags -- description of image features (e.g.,
clear_primary, clear_cloudy_primary, etc.)

### Image Directories
-   **test-jpg** -- approximately 40,000 test images\
-   **train-jpg** -- approximately 40,000 training images\
-   **test-jpg-additional** -- approximately 20,500 additional test
    images

## Technologies
-   Python 3.11
-   PyTorch
-   ResNet50

## Data Loading for the Training Model
```python
# import required libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, multilabel_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Dataset paths configuration
DATA_DIR = './nikitarom/planets-dataset/versions/3/'
TRAIN_DIR = os.path.join(DATA_DIR, 'planet/planet/train-jpg')
TEST_DIR = os.path.join(DATA_DIR, 'planet/planet/test-jpg')
TRAIN_CLASSES = os.path.join(DATA_DIR, 'planet/planet/train_classes.csv')
SUBMISSION = os.path.join(DATA_DIR, 'planet/planet/sample_submission.csv')

# Load CSV files
train_df = pd.read_csv(TRAIN_CLASSES)
submission_df = pd.read_csv(SUBMISSION)
```

I imported the necessary libraries for data processing, set up paths to
training and testing images, and loaded the CSV files.

## Exploring Dataset Information
I displayed:
-   Number of training and test records
-   Table structure and column information
-   Null values
-   Distribution of tags

```python
print(f"Number of training samples: {len(train_df)}")
print(f"Number of test samples: {len(submission_df)}")

print("\\nTraining dataframe preview:")
print(train_df.head())
print("\\nTraining dataframe info:")
print(train_df.info())
print("\\nNull values:")
print(train_df.isnull().sum())

# Extract all tags
all_tags = []
for tags in train_df['tags'].values:
    all_tags.extend(tags.split())
unique_tags = sorted(list(set(all_tags)))
print(f"\\nNumber of unique tags: {len(unique_tags)}")
print(f"Unique tags: {unique_tags}")
```

![image](https://github.com/user-attachments/assets/b7fbb05d-95e6-4d8a-a7f0-8462262d6b22)

I found that there are 17 unique tags describing the image environment:

agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear,
cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy,
primary, road, selective_logging, slash_burn, water

## Tag Frequency Analysis
I analyzed how frequently each tag appears in the dataset and visualized
the distribution using a graph.

![image](https://github.com/user-attachments/assets/cce47905-bde1-4714-a80e-32a0beb21b59)

### Distribution graph of tags
![tags_distribution](https://github.com/user-attachments/assets/652ff75c-a222-46e7-a43c-29773f624a26)

# Model Training and Classifier Creation
## One-Hot Encoding for Tags
To train the model, tags must be converted into vectors.

Each image is represented by a 17-dimensional float vector where:
-   1.0 = tag present
-   0.0 = tag not present

```python
# Convert tag strings into one-hot encoded vectors
def get_tag_map(tags):
    labels = np.zeros(len(unique_tags))
    if pd.isna(tags):
        return labels
    for tag in tags.split():
        if tag in unique_tags:
            labels[unique_tags.index(tag)] = 1
    return labels
# Add encoded vectors to dataframe
train_df['tag_vector'] = train_df['tags'].apply(get_tag_map)
print(train_df.head())
```
This format is required for multi-label classification in PyTorch.

![image](https://github.com/user-attachments/assets/3c2fc5bc-d3e0-4f06-ae01-ac8fb5161937)

## Creating a Custom PyTorch Dataset Class
I created a custom `PlanetDataset` class that:
-   Loads images
-   Applies transformations
-   Converts tag vectors into FloatTensor format
-   Returns image and label vectors

```python
class PlanetDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)
        # Convert tag vector to FloatTensor
        tag_vector = torch.FloatTensor(self.dataframe.iloc[idx]['tag_vector'])
        return image, tag_vector
```
## Image Transformations define
Defined separate transformations for training and validation data:
-   Resize to 224x224
-   Random horizontal and vertical flips
-   Random rotation
-   Color jitter (brightness, contrast, saturation, hue)
-   Normalization using ImageNet mean and standard deviation

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Train / Validation Split
Split dataset into:
-   80% training
-   20% validation

```python
train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)
print(f"\nCount of training samples: {len(train_data)}")
print(f"Count of validation samples: {len(valid_data)}")
# Datasets create
train_dataset = PlanetDataset(train_data, TRAIN_DIR, transform=train_transform)
valid_dataset = PlanetDataset(valid_data, TRAIN_DIR, transform=val_transform)
```

Created PyTorch datasets and DataLoaders with batch size 32.

# Using a Pretrained Model
Created a multi-label classifier based on pretrained ResNet50:
-   First layers partially frozen
-   Replaced final fully connected layer
-   Added custom classifier with:
    -   Linear layers
    -   ReLU activation
    -   Dropout
    -   Final output layer for 17 classes

```python
class PlanetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlanetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Freeze most layers
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
        # Replace final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)
```

# Training Function
The `train_model()` function:
1.  Detects CUDA or CPU
2.  Performs training phase:
    -   Forward pass
    -   Loss calculation
    -   Backpropagation
    -   Optimizer step
3.  Performs validation phase:
    -   No gradient computation
    -   Calculates loss
    -   Converts outputs using sigmoid
    -   Computes precision, recall, and F1 scores
4.  Saves the best model (`best_planet_classifier.pth`)
5.  Visualizes training and validation loss

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        sample_f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        print(f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")
        print(f"Sample F1: {sample_f1:.4f}")
        # Learning rate actualisation
        scheduler.step(epoch_val_loss)
        # Best model save
        if sample_f1 > best_val_f1:
            best_val_f1 = sample_f1
            torch.save(model.state_dict(),
                       'best_planet_classifier.pth')
            print("Best model saved!")
    # Training results visualisation
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Trénovací ztráta')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validační ztráta')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Trénovací a validační ztráta')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_results.png')
    return model
```

**Training model results visualisation**

<img width="4470" height="1466" alt="obrazek" src="https://github.com/user-attachments/assets/ee1c9832-0ef6-4f7e-98ff-b9635e990cad" />

# Model Training
Model components:
-   Loss function: BCEWithLogitsLoss (multi-label classification)
-   Optimizer: Adam
-   Scheduler: ReduceLROnPlateau
-   Number of epochs: 15

After training, the best model is loaded for evaluation.

```python
# Model, criterion, optimiser a scheduler initialisation
model = PlanetClassifier(num_classes=len(unique_tags))
criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label clasification
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

print("Model training...")
num_epochs = 15
start_time = time.time()
trained_model = train_model(model=model, train_loader=train_loader, val_loader=valid_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs)
end_time = time.time()
print(f"Time needed for model train in {num_epochs} epochách: ", end_time - start_time)
```

### Best model evaluation
```python
model.load_state_dict(torch.load('best_planet_classifier.pth'))
```

# Creating Submission File
Generated predictions for test data:
-   Applied sigmoid threshold (0.5)
-   Converted predictions back to tag strings
-   Created submission.csv file

# Confusion Matrices and Evaluation
Evaluated the best model using:
-   Multilabel confusion matrices
-   Average confusion matrix per sample
-   Per-tag metrics:
    -   Accuracy
    -   Precision
    -   Recall
    -   F1 Score

<img width="2280" height="1769" alt="obrazek" src="https://github.com/user-attachments/assets/6bdb26b3-c00c-4c81-bc38-4cc12592c12f" />

<img width="4769" height="3569" alt="obrazek" src="https://github.com/user-attachments/assets/9d277136-5109-4f43-adfb-8741124ba6de" />

<img width="1200" height="509" alt="obrazek" src="https://github.com/user-attachments/assets/a4955ee8-d31d-490a-b8cb-135dce9f44dc" />

Visualized:
-   Confusion matrices for selected tags
-   Average confusion matrix
-   F1 score for all tags

# Summary
This project implements a complete multi-label classification pipeline:
-   Data preprocessing
-   Feature engineering
-   Transfer learning with ResNet50
-   Model training and validation
-   Metric evaluation
-   Submission file generation

The model is designed for satellite image classification of Amazon
rainforest terrain.
