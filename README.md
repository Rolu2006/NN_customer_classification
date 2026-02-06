# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="1024" height="885" alt="Screenshot 2026-02-06 211613" src="https://github.com/user-attachments/assets/b1d2901b-5093-4c6a-9a4c-125933a6ec5d" />








## DESIGN STEPS

STEP 1: Import necessary libraries and load the dataset.

STEP 2:
Encode categorical variables and normalize numerical features.

STEP 3:
Split the dataset into training and testing subsets.

STEP 4:
Design a multi-layer neural network with appropriate activation functions.

STEP 5:
Train the model using an optimizer and loss function.

STEP 6:
Evaluate the model and generate a confusion matrix.

STEP 7:
Use the trained model to classify new data samples.

STEP 8:
Display the confusion matrix, classification report, and predictions.


## PROGRAM

### Name: Somalaraju Rohini
### Register Number: 212224240156

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("customers.csv")
print("Dataset Preview:\n", dataset.head())

# -----------------------------
# Separate features & target
# -----------------------------
X = dataset.drop("Segmentation", axis=1)
y = dataset["Segmentation"]

# -----------------------------
# Handle missing values
# -----------------------------
X["Work_Experience"].fillna(X["Work_Experience"].median(), inplace=True)
X["Family_Size"].fillna(X["Family_Size"].median(), inplace=True)

# -----------------------------
# Encode categorical columns
# -----------------------------
cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# -----------------------------
# Encode target
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Convert to tensors
# -----------------------------
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# -----------------------------
# DataLoader (faster settings)
# -----------------------------
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# Neural Network Model
# -----------------------------
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PeopleClassifier(X.shape[1], len(label_encoder.classes_))

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Training (reduced epochs)
# -----------------------------
epochs = 20
for epoch in range(epochs):
    for xb, yb in loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs} completed")

print("\nTraining Completed")

# -----------------------------
# Evaluation (same data)
# -----------------------------
model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(X), dim=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    target_names=label_encoder.classes_,
    zero_division=0
))

# -----------------------------
# Sample Prediction
# -----------------------------
sample = X[0].unsqueeze(0)
with torch.no_grad():
    pred = model(sample)
    result = label_encoder.inverse_transform([torch.argmax(pred).item()])

print("\nSample Prediction:", result[0])
        

```
```python
# Initialize the Model, Loss Function, and Optimizer


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here
```



## Dataset Information
<img width="1023" height="315" alt="Screenshot 2026-02-06 210815" src="https://github.com/user-attachments/assets/9b9103b6-7f7a-480c-8791-89554100650a" />







## OUTPUT



### Confusion Matrix




<img width="555" height="630" alt="Screenshot 2026-02-06 210835" src="https://github.com/user-attachments/assets/da86efa1-bb24-48cc-af3b-5f15c3fa1660" />



### Classification Report




<img width="642" height="260" alt="Screenshot 2026-02-06 210848" src="https://github.com/user-attachments/assets/ab2408f4-e863-46c9-a961-cc81e917b97f" />


### New Sample Data Prediction




<img width="783" height="223" alt="Screenshot 2026-02-06 210856" src="https://github.com/user-attachments/assets/d066d934-0ebb-4006-a05a-e7ca4671c408" />





<img width="533" height="61" alt="Screenshot 2026-02-06 210902" src="https://github.com/user-attachments/assets/cc159b13-97d0-4a69-baa6-11bf8c489331" />



## RESULT
Include your result here
