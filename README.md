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

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # 4 output classes (A, B, C, D)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)  # No softmax (handled by CrossEntropyLoss)
        return x
```
```
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```
```
input_size = X_train.shape[1]

model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Dataset Information


<img width="1358" height="694" alt="Screenshot 2026-03-17 183645" src="https://github.com/user-attachments/assets/09c46cf1-c0e7-4eb0-8d0a-4f1761c5b728" />





## OUTPUT



### Confusion Matrix




<img width="986" height="613" alt="Screenshot 2026-03-17 183734" src="https://github.com/user-attachments/assets/00fb5816-49b0-4d73-8dcb-5d9567ee5ba3" />




### Classification Report


<img width="1063" height="592" alt="Screenshot 2026-03-17 183806" src="https://github.com/user-attachments/assets/d96133d1-005a-4cb3-ae33-32ddb1423d6a" />





### New Sample Data Prediction




<img width="1196" height="442" alt="Screenshot 2026-03-17 183857" src="https://github.com/user-attachments/assets/657bab41-2cf5-4513-9b68-e36ceab72a30" />



## RESULT
The neural network model was successfully trained on the customer dataset and achieved a good classification performance, producing accurate predictions for customer segmentation (A, B, C, D) with evaluated metrics such as accuracy, confusion matrix, and classification report.
