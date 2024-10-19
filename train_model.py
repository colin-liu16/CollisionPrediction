from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


def evaluate(model, data_loader, loss_function):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for sample in data_loader:
            inputs = sample['input']
            labels = sample['label']
            outputs = model(inputs)
            labels = labels.view_as(outputs)

            loss = loss_function(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = outputs > 0.5
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Early stopping parameters
    patience = 12
    epochs_since_improvement = 0
    best_test_loss = float('inf')
    os.makedirs('saved', exist_ok=True)

    for epoch_i in range(no_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for idx, sample in enumerate(data_loaders.train_loader):
            inputs = sample['input']
            labels = sample['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.view_as(outputs)  # Ensure labels have the same shape as outputs

            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)  # Multiply by batch size

            preds = outputs > 0.5
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

        epoch_train_loss = running_loss / len(data_loaders.train_loader.dataset)
        epoch_train_accuracy = total_correct / total_samples
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Evaluation phase
        test_loss, test_accuracy = evaluate(model, data_loaders.test_loader, loss_function)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f'Epoch [{epoch_i + 1}/{no_epochs}], Training Loss: {epoch_train_loss:.4f}, '
              f'Training Acc: {epoch_train_accuracy:.4f}, Test Loss: {test_loss:.4f}, '
              f'Test Acc: {test_accuracy:.4f}')

        # Check for improvement
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_since_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), 'saved/saved_model.pkl')
            print(f"Test loss decreased to {best_test_loss:.4f}. Saving model...")
        else:
            epochs_since_improvement += 1
            print(f"No improvement in test loss for {epochs_since_improvement} epoch(s).")

        # Early stopping condition
        if epochs_since_improvement >= patience:
            print(f"Early stopping after {epoch_i + 1} epochs.")
            break

    epochs = range(1, epoch_i + 2)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig('training_testing_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.savefig('training_testing_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    no_epochs = 10000
    train_model(no_epochs)
