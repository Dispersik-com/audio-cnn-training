import os
import torch
from net import CNN, train_model, test_model
from data_utils import load_data, split_data
import config


def main():
    cnn_net = CNN(num_classes=config.num_classes, dropout_prob=config.dropout_prob)

    data_loader = load_data(config.file_paths)
    train_data, test_data = split_data(data_loader, size=0.8)

    train_model(cnn_net, train_data, config.num_epochs,
                config.learning_rate, config.show_plots)

    test_model(cnn_net, test_data)

    if config.load_and_continue and os.path.exists(config.model_load_path):
        cnn_net = torch.load(config.model_load_path)
        cnn_net.load_state_dict(torch.load(config.model_load_path))
        print(f"Loaded model from {config.model_load_path}")

    if config.test_network:
        test_model(cnn_net, test_data)

    if config.save_model:
        torch.save(cnn_net.state_dict(), config.model_save_path)
        print(f"Model saved to {config.model_save_path}")


if __name__ == "__main__":
    main()
