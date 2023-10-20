import torch
import torch.nn as nn
import torch.optim as optim

# transformer for 1d time series prediction
# https://www.datasciencebyexample.com/2023/05/15/stock-price-prediction-using-transformer-in-pytorch/
class SinusTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(SinusTransformer, self).__init__()
        # input/output linear to put things into shape for transformer
        # because we need d_model == d_src == d_tgt
        self.input_linear = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.input_linear(src)
        tgt = self.input_linear(tgt)
        output = self.transformer(src, tgt)
        output = self.output_linear(output)
        return output
    

# sinus wave data
# 512 steps betweewn 0 and 10
t = torch.linspace(0, 10, 512)
sin_wave = torch.sin(t)  

# I don't really know whats good for what
d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1

# model, loss, optimizer
# the paper also uses Adam
model = SinusTransformer(d_model, nhead, num_layers, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# use 32 values to predict the next 1 value
input_size = 32
output_size = 1
num_epochs = 6

# accumulate losses/predictions
# don't need to split train/test because its periodic data
all_losses = []
all_predictions = []

for epoch in range(num_epochs):

    epoch_losses = []
    epoch_predictions = []
    
    for i in range(len(sin_wave) - input_size - output_size):

        # reshape input/output but I don't know why
        X = sin_wave[i:i+input_size].view(input_size, 1, 1)
        y = sin_wave[i+input_size:i+input_size+output_size].view(output_size, 1, 1)

        optimizer.zero_grad()
        out = model(X, y)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # accumulate losses/predictions
        epoch_losses.append(loss.item())
        epoch_predictions.append(out.view(-1).detach().numpy())

    all_losses.append(epoch_losses)
    all_predictions.append(epoch_predictions)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


import matplotlib.pyplot as plt

# group losses and predictions by epochs
losses = list(map(lambda xs: sum(xs)/len(xs), zip(*all_losses)))
predictions = list(map(lambda xs: sum(xs)/len(xs), zip(*all_predictions)))

# plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Average Loss')
plt.title('Training Loss')
plt.xlabel('Iteration per Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.savefig('avg_loss.png')

# plot the predictions
plt.figure(figsize=(10, 5))
plt.plot(sin_wave[input_size+1:], label='Ground Truth', color='blue')
plt.plot(predictions, label='Average Predictions', color='red')
plt.title('Ground Truth vs. Average Predictions')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.savefig('avg_prediction.png')
