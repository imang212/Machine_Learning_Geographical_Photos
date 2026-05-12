<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-ResNet50-blue?style=flat" />
  <img src="https://img.shields.io/badge/Task-Multi--Label%20Classification-orange?style=flat" />
</p>

# Satellite Image Classification for Geographic Data
**Amazon Rainforest Classification** is a deep learning project focused on multi-label classification of satellite imagery. Using the **Planet: Understanding the Amazon from Space** dataset, the model identifies 17 unique tags covering atmospheric conditions, land cover, and land use.

## Technical Highlights
Unlike standard image classification, this project implements advanced deep learning techniques:
- **Custom Normalization:** Mean and Std calculated specifically for this dataset ([0.3117, 0.3408, 0.2991]).
- **Focal Loss:** Implemented to handle high class imbalance (focusing on "hard" rare samples).
- **Global Average Pooling (GAP):** Replaced standard pooling to improve spatial robustness and reduce overfitting.
- **Transfer Learning:** Fine-tuned ResNet50 with custom classification heads and AdamW optimization.

## Dataset
The dataset contains approximately 60,000 test images and 40,000 training images of
Amazon rainforest terrain. It also includes CSV files describing both
the training and test images.

- Multi-Label: Each image can have multiple tags (e.g., primary, water, agriculture).
- 17 Unique Tags: Ranging from clear sky to specific land uses like slash_burn or artisinal_mine.

The dataset can be freely used for machine learning under the Database
Contents License.

**Dataset link:**

https://www.kaggle.com/datasets/nikitarom/planets-dataset/data

**The root structure of the dataset:**

![image](https://github.com/user-attachments/assets/1fab1cc9-de39-4a88-a92f-6ae9f64b1def)

### CSV Files
**sample_submission.csv** - 61,191 records - 2 columns: - image_name --
name of the image - tags -- description of features present in the image

**train_classes.csv** - 40,479 records - 2 columns: - image_name -- name
of the image - tags -- description of image features (e.g.,
clear_primary, clear_cloudy_primary, etc.)

### Image Directories
-   **test-jpg** -- approximately 40,000 test images
-   **train-jpg** -- approximately 40,000 training images
-   **test-jpg-additional** -- approximately 20,500 additional test
    images

## Technologies
-   Python 3.11
-   PyTorch
-   ResNet50

## Model Architecture (ResNet50 + GAP)
The model utilizes a pre-trained ResNet50 backbone with a custom head optimized for satellite data:
1. Backbone: ResNet50 (pre-trained on ImageNet).
2. Pooling: AdaptiveAvgPool2d (Global Average Pooling).
3. Classifier: - Linear(2048 -> 512) -> BatchNorm -> ReLU -> Dropout(0.2)
    - Final Linear layer for 17 classes.

## Installation & Usage
1. Clone the repo:
```Bash
git clone [https://github.com/imang212/ML_Satellite_Image_Classification.git](https://github.com/imang212/ML_Satellite_Image_Classification.git)
```
2. Install dependencies:
```Bash
pip install torch torchvision pandas numpy seaborn matplotlib scikit-learn Pillow
```
3. Run the Notebook:
Open `train.ipynb` in Jupyter or VS Code to see the full training and evaluation process.

## Data Loading for the Training Model
```python
# import of required datasets
import os
import pandas as pd
import numpy as np
import seaborn as sns
import time
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, multilabel_confusion_matrix, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Setting paths to data
DATA_DIR = './nikitarom/planets-dataset/versions/3/'
TRAIN_DIR = os.path.join(DATA_DIR, 'planet/planet/train-jpg')
TEST_DIR = os.path.join(DATA_DIR, 'planet/planet/test-jpg')
TRAIN_CLASSES = os.path.join(DATA_DIR, 'planet/planet/train_classes.csv')
SUBMISSION = os.path.join(DATA_DIR, 'planet/planet/sample_submission.csv')

# Loading CSV files
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

# Display information about the table
print('\nDisplaying information about the training table:')
print("Table header: \n", train_df.head(), '\n')
print("Table info: ", train_df.info(), '\n')
print("Null values: ", train_df.isnull().sum(), '\n')

# Display the tag distribution in the 'tags' column
all_tags = []
for tags in train_df['tags'].values:
    all_tags.extend(tags.split())
unique_tags = sorted(list(set(all_tags)))
print(f"\nNumber of unique tags: {len(unique_tags)}")
print(f"Unique tags: {unique_tags}")
```

<img width="817" height="579" alt="Unique_tags_print" src="https://github.com/user-attachments/assets/bd6eab3d-622b-47ff-9224-58dd106ae0b7" />

I found that there are 17 unique tags describing the image environment:

agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear,
cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy,
primary, road, selective_logging, slash_burn, water

## Tag Frequency Analysis
I analyzed how frequently each tag appears in the dataset and visualized
the distribution using a graph.

<img width="318" height="229" alt="Tags_distribution_print" src="https://github.com/user-attachments/assets/a105264d-0c70-426a-917a-b935a0bb75a3" />

### Distribution graph of tags

<img width="1200" height="500" alt="tags_distribution_train" src="https://github.com/user-attachments/assets/c994a561-4d78-4f9e-b9a5-1414d334ffaf" />

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
print(f"\nNumber of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(valid_data)}")
# Creating datasets
train_dataset = PlanetDataset(train_data, TRAIN_DIR, transform=train_transform)
valid_dataset = PlanetDataset(valid_data, TRAIN_DIR, transform=val_transform)
```

<img width="398" height="52" alt="Snímek obrazovky z 2026-05-12 17-43-44" src="https://github.com/user-attachments/assets/d3f8ab92-d030-487b-955d-4dfbd69d898d" />

Created PyTorch datasets and DataLoaders with batch size 32.

```python
# Dataloaders create
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
```

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
# Creating a multi-label classifier using ResNet50
class PlanetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlanetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in list(self.resnet.parameters())[:-10]: param.requires_grad = False # only for smaller datasets
        # Remove original avgpool and fc layers
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        # Global Average Pooling instead of standard
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # [batch, 2048, 2, 2] last layer of ResNet50, reduces output to 2x2
        #in_features = self.resnet.fc.in_features # Replacing the final fully connected layer for multi-label classification
        spatial_features = 2048 # 2048 channels, each with 2x2 spatial features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)  # [batch, 2048, H, W], Input image x passes through the backbone network, typically a pre-trained CNN (e.g. ResNet, VGG)
        pooled = self.global_avg_pool(features)  # [batch, 2048, 1, 1]
        output = self.classifier(pooled)  # [batch, num_classes], applying the sequence
        return output
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, model_name='ResNet50', device=None):
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    best_val_f1 = 0.0
    train_losses = []; val_losses = []; train_F1_scores = []; val_F1_scores = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        model.train()
        running_loss = 0.0; all_train_preds = []; all_train_labels = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            # for multilabel classification
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_train_preds.append(preds.cpu())
            all_train_labels.append(labels.cpu())
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        all_train_preds, all_train_labels = torch.cat(all_train_preds, dim=0).numpy(), torch.cat(all_train_labels, dim=0).numpy()
        train_f1 = f1_score(all_train_labels, all_train_preds, average='samples', zero_division=0)
        train_F1_scores.append(train_f1)

        model.eval()
        running_loss = 0.0; all_preds = []; all_labels = []
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
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        all_preds, all_labels = torch.cat(all_preds, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()
        # Calculating metrics
        precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
        sample_f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_F1_scores.append(sample_f1)
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {sample_f1:.4f}')
        print(f'Precision: {precision:.4f}, Real Precision(Recall): {recall:.4f}')
        print(f'Sample F1(Harmonic mean): {sample_f1:.4f}, Macro F1(Score for every class): {macro_f1:.4f}')
        tag_f1_scores = []
        for i, tag in enumerate(unique_tags):
            tag_f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            tag_f1_scores.append((tag, tag_f1))
        tag_f1_scores.sort(key=lambda x: x[1], reverse=True)
        print("\nF1 scores for the best tags:")
        for tag, f1 in tag_f1_scores[:5]:
            print(f"{tag}: {f1:.4f}")
        print("\nF1 scores for the worst tags:")
        for tag, f1 in tag_f1_scores[-10:]:
            print(f"{tag}: {f1:.4f}")

        # Updating learning rate
        #scheduler.step()
        scheduler.step(epoch_val_loss)

        if sample_f1 > best_val_f1:
            best_val_f1 = sample_f1
            torch.save(model.state_dict(), f"best_planet_classifier_{model_name}.pth")
            print("Best model saved!")
    # Visualizing training results
    plt.figure(figsize=(15, 5))
    # Loss chart
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss (%)'); plt.title('Training and validation loss')
    plt.grid(True)
    plt.legend()
    # Accuracy chart
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_F1_scores, 'g-', label='Training F1 score')
    plt.plot(range(1, num_epochs+1), val_F1_scores, 'm-', label='Validation F1 score')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Training and validation F1 score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout(); plt.savefig('training_results.png', dpi=300, bbox_inches='tight'); plt.show()
    return model
```

**Training model results visualisation**

<img width="4471" height="1466" alt="training_results" src="https://github.com/user-attachments/assets/fd2d6a60-66cc-45fa-bdc6-fef358b99c58" />

# Model Training
Model components:
-   Loss function: BCEWithLogitsLoss (multi-label classification)
-   Optimizer: Adam
-   Scheduler: ReduceLROnPlateau
-   Number of epochs: 15

After training, the best model is loaded for evaluation.

```python
# Initializing model, criterion, optimizer, and scheduler
num_epochs = 15
model = PlanetClassifier(num_classes=len(unique_tags)) # ResNet50 model
#criterion = nn.BCEWithLogitsLoss() #pos_weight=class_weights
criterion = FocalLoss(alpha=1, gamma=2, pos_weight=None) # Focal loss for class balancing and focusing on hard cases
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)) # AdamW is an improved version of Adam with weight regularization
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-6, last_epoch=-1)

print("Training the model...")
start_time = time.time()
trained_model = train_model(model=model, train_loader=train_loader, val_loader=valid_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs, model_name="ResNet50")
end_time = time.time()
print(f"Time to train the model in {num_epochs} epochs: ", (end_time - start_time)/60, " minutes, ", (end_time - start_time)/60/60, " hours")
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

```python
def create_submission(model, test_loader, max_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []; image_names = []; processed_samples = 0
    with torch.no_grad():
        for batch, (inputs, _) in enumerate(test_loader):
            if processed_samples >=max_samples: break
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            for i in range(len(preds)):
                idx = batch * test_loader.batch_size + i
                if processed_samples < max_samples:
                    img_name = submission_df.iloc[idx]['image_name']
                    image_names.append(img_name)
                    # Getting tags for prediction
                    pred_tags = []
                    for j, val in enumerate(preds[i]):
                        if val == 1:
                            pred_tags.append(unique_tags[j])
                    predictions.append(' '.join(pred_tags))
                    processed_samples += 1
    print(f"Processed {processed_samples} samples.")
    # Creating submission dataframe
    submit_df = pd.DataFrame({'image_name': image_names, 'tags': predictions })
    # Saving to CSV
    submit_df.to_csv('submission.csv', index=False)
    print("Submission file has been created!")

# loading the test dataset
filtered_submission_df = submission_df.head(10000)
filtered_submission_df['tag_vector'] = [np.zeros(len(unique_tags)) for _ in range(len(filtered_submission_df))]
test_dataset = PlanetDataset(filtered_submission_df, TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

start_time = time.time()
create_submission(trained_model, test_loader, max_samples=10000)
end_time = time.time()
print(f"Time to create submission on test data: ", (end_time - start_time)/60, " minutes, ", (end_time - start_time)/60/60, " hours")
```

# Confusion Matrices and Evaluation
Evaluated the best model using:
-   Multilabel confusion matrices
-   Average confusion matrix per sample
-   Per-tag metrics:
    -   Accuracy
    -   Precision
    -   Recall
    -   F1 Score
      
<img width="398" height="52" alt="all_tags_confusion_matrix" src="https://github.com/user-attachments/assets/1a7e53cc-7e5f-413b-9dff-c4bdbc6c746b" />

<img width="2306" height="1769" alt="aggregated_confusion_matrix" src="https://github.com/user-attachments/assets/4fc971ac-ff94-421d-bd43-33ca9003705d" />

<img width="1200" height="509" alt="obrazek" src="https://github.com/user-attachments/assets/a4955ee8-d31d-490a-b8cb-135dce9f44dc" />

Visualized:
-   Confusion matrices for selected tags
-   Average confusion matrix
-   F1 score for all tags

# Tags prediction (on submission data)
<img width="1992" height="1201" alt="tags_prediction" src="https://github.com/user-attachments/assets/174f66a3-2b0c-40e2-b049-e8874f0c2e5e" />

