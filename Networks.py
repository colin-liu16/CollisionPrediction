import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super(Action_Conditioned_FF, self).__init__()
        # Define the network architecture
        self.fc1 = nn.Linear(6, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Input validation and reshaping
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)
        
        # Ensure input is 2D: [batch_size, features]
        if input.dim() == 1:
            input = input.unsqueeze(0)  
        
        if input.size(-1) != 6:
            raise ValueError(f"Expected 6 input features, got {input.size(-1)}")
            
        try:
            x = self.fc1(input)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            output = self.sigmoid(x)
            
            if output.dim() > 1:
                output = output.squeeze(-1)
            return output
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.zeros(input.size(0))

    def evaluate(self, model, test_loader, loss_function):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input']
                labels = batch['label']
                
                outputs = model(inputs)
                
                # Ensure labels match output dimensions
                if labels.dim() != outputs.dim():
                    labels = labels.view_as(outputs)
                
                loss = loss_function(outputs, labels)
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        average_loss = total_loss / total_samples if total_samples > 0 else 0
        return average_loss

def main():
    model = Action_Conditioned_FF()
    
    # Test the model with sample data
    sample_input = torch.randn(4, 6)  # Batch of 4 samples, 6 features each
    output = model(sample_input)
    print(f"Sample output shape: {output.shape}")

if __name__ == '__main__':
    main()
