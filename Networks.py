import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()

        # Define the network architecture
        self.fc1 = nn.Linear(6, 32)  # First fully connected layer with 32 units
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)  # Second fully connected layer with 16 units
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        try:
            x = self.fc1(input)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            output = self.sigmoid(x)

            return output.squeeze(dim=1)
        except TypeError as e:
            print(f"TypeError in forward pass: {e}")
            if isinstance(input, torch.Tensor):
                batch_size = input.size(0)
            else:
                batch_size = 1
            return torch.zeros(batch_size)


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input']
                labels = batch['label']
                outputs = model(inputs)
                labels = labels.view_as(outputs)
                loss = loss_function(outputs, labels)
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        if total_samples:
            average_loss = total_loss / total_samples
        else:
            average_loss = 0
        return average_loss


def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
