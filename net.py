import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt


class CNN(nn.Module):
    def __init__(self, num_classes, dropout_prob=None):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.dropout_prob = dropout_prob
        self.fc = nn.Linear(64 * 5 * 5, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.dropout_prob is not None:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        if self.dropout_prob is not None:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.output(x)
        return x


def train_model(model, train_loader, num_epochs=100, learning_rate=0.001, show_plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7)

    accuracies = []
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs).squeeze(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Update the learning rate
        scheduler.step(epoch_loss)

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy over Epochs')
        plt.legend()

        plt.show()


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct_predictions = 0
    total_samples = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs).squeeze(0)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_loss /= total_samples
    test_accuracy = correct_predictions / total_samples

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    #
    # for inputs, labels in test_loader:
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     inputs = inputs.unsqueeze(1)
    #     outputs = model(inputs).squeeze(0)
    #     probabilities = F.softmax(outputs, dim=1)
    #     print(probabilities)
