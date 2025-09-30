# Stock-Price-Prediction
## AIM
To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Given dataset of stock prices with relevant features. Extract one column and perform prediction for the next data in the series using Recurrent Neural Network. 

## Design Steps
### Step 1:
Import required libraries and load the dataset separately for training and testing.
### Step 2:
Preprocess the data using MinMaxScaler. Create sequences and split into x train, y train and x test, y test. Convert to tensors and create dataloader for batchwise training.
### Step 3:
Define the recurrent model and run it for the dataset.
### Step 4: 
Display the results.

## Program
#### Name: Cynthia Mehul J
#### Register Number: 212223240020

```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self,input_size=1, hidden_size=32, num_layers=4, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
  train_losses = []
  for epoch in range(num_epochs):
      model.train()
      epoch_loss = 0
      for x_batch, y_batch in train_loader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)

          optimizer.zero_grad()
          outputs = model(x_batch)
          loss = criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()

          epoch_loss += loss.item()

      epoch_loss /= len(train_loader)
      train_losses.append(epoch_loss)
      if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
  return train_losses

```

## Output

### True Stock Price, Predicted Stock Price vs time and Predictions

<img width="891" height="621" alt="image" src="https://github.com/user-attachments/assets/7eb82502-97ea-4dc7-8101-4177e9766699" />

### Training Loss Curve

<img width="604" height="506" alt="image" src="https://github.com/user-attachments/assets/b61e6fe4-da6c-49b7-9da0-a5ea26801fcd" />

## Result
Therefore, recurrent neural network model is developed and executed for stock price prediction successfully.
