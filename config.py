# config.py

# Model parameters
num_classes = 6
dropout_prob = None  # If None model will be training without dropout

# Train parameters
num_epochs = 100
learning_rate = 1e-4
batch_size = 32
num_workers = 3
train_split = 0.8

# data paths
path_prefix = "dataset v37/pths"
file_paths = [
    # paths, classes
    (f"{path_prefix}/string 1 (noise).pth", 0),
    (f"{path_prefix}/string 2 (noise).pth", 1),
    (f"{path_prefix}/string 3 (noise).pth", 2),
    (f"{path_prefix}/string 4 (noise).pth", 3),
    (f"{path_prefix}/string 5 (noise).pth", 4),
    (f"{path_prefix}/string 6 (noise).pth", 5),

    (f"{path_prefix}/string 1 (clean).pth", 0),
    (f"{path_prefix}/string 2 (clean).pth", 1),
    (f"{path_prefix}/string 3 (clean).pth", 2),
    (f"{path_prefix}/string 4 (clean).pth", 4),
    (f"{path_prefix}/string 5 (clean).pth", 4),
    (f"{path_prefix}/string 6 (clean).pth", 5)
]

# Launch parameters
test_network = True  # If True, the network will be tested after training
show_plots = True  # If True, training plots will be displayed
load_and_continue = False  # If True, a saved model will be loaded for continued training
save_model = True  # If True, the trained model will be saved to disk
model_save_path = "model_v1_1.pth"  # Path to save the model
model_load_path = "model_v1_1.pth"  # Path to load the model
