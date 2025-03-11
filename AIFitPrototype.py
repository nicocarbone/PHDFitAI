import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import random
import time
import json

import os

class TurbidMediaDataset(Dataset):
    def __init__(self, histograms, irfs, ups, ua):
        self.histograms = histograms
        self.irfs = irfs
        self.ups = ups
        self.ua = ua

    def __len__(self):
        return len(self.histograms)

    def __getitem__(self, idx):
        histogram = torch.tensor(self.histograms[idx], dtype=torch.float32).unsqueeze(0) # Add channel dimension
        irf = torch.tensor(self.irfs[idx], dtype=torch.float32).unsqueeze(0) # Add channel dimension
        ups = torch.tensor(self.ups[idx], dtype=torch.float32)
        ua = torch.tensor(self.ua[idx], dtype=torch.float32)
        return histogram, irf, ups, ua

# Custom Exp Layer Module
class ExpLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)

class OpticalPropertiesEstimator(nn.Module):
    def __init__(self, config): # Config dictionary as input
        super(OpticalPropertiesEstimator, self).__init__()
        self.config = config # Store config

        # CNN branch for Histogram
        hist_cnn_layers = []
        in_channels = 1 # Input channels for the first CNN layer
        cnn_filters_hist = config.get('cnn_filters_hist', [16, 32, 64]) # Convolutional neural network filter sizes
        kernel_size_cnn = config.get('kernel_size_cnn', 3) # Convolutional kernel size
        pool_size_cnn = config.get('pool_size_cnn', 2) # Pooling size
        padding_cnn = config.get('padding_cnn', 1) # Padding size
        use_batchnorm_cnn = config.get('use_batchnorm_cnn', True) # Configurable batch norm

        for out_channels in cnn_filters_hist:
            hist_cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size_cnn, padding=padding_cnn)) # 1D Convolutional layer
            if use_batchnorm_cnn:
                hist_cnn_layers.append(nn.BatchNorm1d(out_channels)) # Batch norm layer
            hist_cnn_layers.append(nn.ReLU())  # ReLU activation
            hist_cnn_layers.append(nn.MaxPool1d(pool_size_cnn))  # Max pooling
            in_channels = out_channels
        self.hist_cnn = nn.Sequential(*hist_cnn_layers)

        # CNN branch for IRF (similar structure to hist_cnn)
        irf_cnn_layers = []
        in_channels = 1
        cnn_filters_irf = config.get('cnn_filters_irf', [16, 32, 64]) # Default filter sizes for IRF CNN
        for out_channels in cnn_filters_irf:
            irf_cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size_cnn, padding=padding_cnn)) # 1D Convolutional layer
            if use_batchnorm_cnn:
                irf_cnn_layers.append(nn.BatchNorm1d(out_channels)) # Batch norm layer
            irf_cnn_layers.append(nn.ReLU()) # ReLU activation
            irf_cnn_layers.append(nn.MaxPool1d(pool_size_cnn)) # Max pooling
            in_channels = out_channels
        self.irf_cnn = nn.Sequential(*irf_cnn_layers)

        # Calculate flattened size after CNN layers by running a tryal forward pass
        example_input = torch.randn(1, 1, 4096) # Dummy input for size calculation
        hist_cnn_output_size = self._get_cnn_output_size(self.hist_cnn, example_input)
        irf_cnn_output_size = self._get_cnn_output_size(self.irf_cnn, example_input)
        combined_cnn_output_size = hist_cnn_output_size + irf_cnn_output_size

        # Fully connected layers with BatchNorm
        fc_layers = []
        fc_layer_sizes = config.get('fc_layer_sizes', [128, 64]) # Default FC layer sizes
        dropout_rate_fc = config.get('dropout_rate_fc', 0.2) # Configurable dropout
        in_features = combined_cnn_output_size

        for layer_size in fc_layer_sizes:
            fc_layers.append(nn.Linear(in_features, layer_size)) # Fully connected layer
            fc_layers.append(nn.BatchNorm1d(layer_size)) # Batch norm layer
            fc_layers.append(nn.ReLU()) # ReLU activation
            fc_layers.append(nn.Dropout(dropout_rate_fc)) # Dropout layer
            in_features = layer_size
        fc_layers.append(nn.Linear(in_features, 2)) # Output layer (ups and ua)
        fc_layers.append(ExpLayer()) # Ensure positive output using Exponential activation
        self.fc = nn.Sequential(*fc_layers)


    def _get_cnn_output_size(self, cnn_layers, input_tensor):
        '''
        Helper function to calculate the output size after custom CNN layers
        '''
        output = cnn_layers(input_tensor)
        return int(np.prod(output.size()[1:])) # Return flattened size

    def forward(self, histogram, irf):
        hist_features = self.hist_cnn(histogram)
        hist_features = hist_features.view(hist_features.size(0), -1) # Flatten
        irf_features = self.irf_cnn(irf)
        irf_features = irf_features.view(irf_features.size(0), -1) # Flatten
        combined_features = torch.cat((hist_features, irf_features), dim=1)
        output = self.fc(combined_features)
        return output

# Custom MSE Loss with UA Weight
class MSELossWeightedUA(nn.Module):
    def __init__(self, ua_weight=1.0): # Add ua_weight as a parameter, default to 1.0
        super().__init__()
        self.ua_weight = ua_weight

    def forward(self, predictions, targets):
        # No need to clamp predictions here as ExpLayer ensures positivity
        # Targets are assumed to be positive or non-negative

        squared_error = (predictions - targets)**2

        # Separate errors for ups and ua
        ups_error = squared_error[:, 0] # Error for ups (first output)
        ua_error = squared_error[:, 1]  # Error for ua (second output)

        # Apply weight to ua error
        weighted_ua_error = self.ua_weight * ua_error

        # Combine the losses (sum and then average)
        loss = torch.mean(ups_error + weighted_ua_error)

        return loss


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, device="cpu", clip_grad=1.0, patience=10): # Added patience for early stopping
    model.to(device)
    train_losses = []
    val_losses = []
    learning_rates = []

    best_val_loss = float('inf') # Initialize best validation loss to infinity
    epochs_no_improve = 0       # Counter for epochs with no improvement
    best_model_state = None      # Store the state of the best model

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for histograms, irfs, ups, ua in train_loader:
            histograms = histograms.to(device)
            irfs = irfs.to(device)
            ups = ups.to(device)
            ua = ua.to(device)

            optimizer.zero_grad()
            outputs = model(histograms, irfs)
            loss = criterion(outputs, torch.stack((ups, ua), dim=1))
            loss.backward()

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for histograms, irfs, ups, ua in val_loader:
                histograms = histograms.to(device)
                irfs = irfs.to(device)
                ups = ups.to(device)
                ua = ua.to(device)
                outputs = model(histograms, irfs)
                val_loss = criterion(outputs, torch.stack((ups, ua), dim=1))
                val_running_loss += val_loss.item()

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}')

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0 # Reset counter
            best_model_state = model.state_dict() # Save the state of the current best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}. Validation loss did not improve for {patience} epochs.')
            model.load_state_dict(best_model_state) # Load the best model state
            break # Stop training

    return model, train_losses, val_losses, learning_rates





if __name__ == '__main__':

    # Iterate through each file in the folder SIMs that starts with sim_results_
    # First extract the suffix from the filename so they are loaded in the same order
    sim_runs = []
    for filename in os.listdir('SIMs'):
        if filename.startswith('sim_results_'):
            suffix = filename.split('_')[-1] # Extract the suffix
            sim_runs.append(suffix)

    histograms_data = []
    for suffix in sim_runs:
        filename = 'sim_results_' + suffix
        file_path = os.path.join('SIMs', filename)
        histograms_data.append(np.load(file_path))

    histograms_data = np.concatenate(histograms_data, axis=0) # Concatenate all histograms
    histograms_data =  np.array(histograms_data, dtype=np.float32) # Convert to float32


    irfs_data = []
    for suffix in sim_runs:
        filename = 'sim_irfs_' + suffix
        file_path = os.path.join('SIMs', filename)
        irfs_data.append(np.load(file_path))

    irfs_data = np.concatenate(irfs_data, axis=0) # Concatenate all IRFs
    irfs_data = np.array(irfs_data, dtype=np.float32) # Convert to float32


    tags_data = []
    for suffix in sim_runs:
        filename = 'sim_tags_' + suffix
        file_path = os.path.join('SIMs', filename)
        tags_data.append(np.load(file_path))

    tags_data = np.concatenate(tags_data, axis=0) # Concatenate all tags
    tags_data = np.array(tags_data, dtype=np.float32) # Convert to float32


    ua_data = tags_data[:, 0] # Extract ups
    ups_data = tags_data[:, 1] # Extract ua

    print(len(ua_data))
    print(ua_data)
    print(ups_data)

    ## 2. Split data into training and validation sets
    histograms_train, histograms_val, irfs_train, irfs_val, ups_train, ups_val, ua_train, ua_val = train_test_split(
        histograms_data, irfs_data, ups_data, ua_data, test_size=0.2, random_state=42
    )

    # 3. Create Datasets and DataLoaders

    batch_size = 64

    train_dataset = TurbidMediaDataset(histograms_train, irfs_train, ups_train, ua_train)
    val_dataset = TurbidMediaDataset(histograms_val, irfs_val, ups_val, ua_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Hyperparameter Search Space (Define ranges/choices for hyperparameters)
    hyperparameter_space = {
        'cnn_filters_hist': [[16, 32, 128, 256, 1024], 
                             [16, 32, 128, 512, 1024], 
                             [32, 128, 256, 1024],
                             [32, 128, 256, 512]], # List of lists for CNN filters
        'cnn_filters_irf': [[16, 32, 128, 256, 1024], 
                            [16, 32, 128, 512, 1024], 
                            [32, 128, 256, 1024],
                            [32, 128, 256, 512]], # List of lists for CNN filters
        'kernel_size_cnn': [5, 10, 15, 20],
        'pool_size_cnn': [2],
        'padding_cnn': [1, 2],
        'use_batchnorm_cnn': [True, False],
        'fc_layer_sizes': [[1024, 512, 256, 128, 32], 
                           [1024, 256, 128, 32], 
                           [512, 256, 128, 32],
                           [256, 128, 32]], # List of lists for FC layers
        'dropout_rate_fc': [0.0, 0.2, 0.4, 0.6] ,
        'lr': [0.002],
        'step_size': [10],
        'gamma': [0.8],
        'batch_size': [batch_size],
        'ua_weight': [300.0],
        'clip_grad': [1.0, None], # Example of including None (no clip_grad)
        'patience': [100] # Example for early stopping patience
    }

    num_trials = 40  # Number of random hyperparameter combinations to try
    trial_epochs = 100  # Number of epochs to train each trial
    best_val_loss = float('inf')
    best_hyperparameters = None
    history = [] # To store results of each trial

    for trial in range(num_trials):
        print(f"\n--- Trial {trial+1}/{num_trials} ---")

        # 5. Sample Hyperparameters Randomly
        current_hyperparameters = {}
        for param_name, param_values in hyperparameter_space.items():
            current_hyperparameters[param_name] = random.choice(param_values)

        print("Current Hyperparameters:", current_hyperparameters)

        # 6. Create Network Configuration from Sampled Hyperparameters
        network_config = {
            'cnn_filters_hist': current_hyperparameters['cnn_filters_hist'],
            'cnn_filters_irf': current_hyperparameters['cnn_filters_irf'],
            'kernel_size_cnn': current_hyperparameters['kernel_size_cnn'],
            'pool_size_cnn': current_hyperparameters['pool_size_cnn'],
            'padding_cnn': current_hyperparameters['padding_cnn'],
            'use_batchnorm_cnn': current_hyperparameters['use_batchnorm_cnn'],
            'fc_layer_sizes': current_hyperparameters['fc_layer_sizes'],
            'dropout_rate_fc': current_hyperparameters['dropout_rate_fc']
        }
        batch_size = current_hyperparameters['batch_size']
        lr = current_hyperparameters['lr']
        step_size = current_hyperparameters['step_size']
        gamma = current_hyperparameters['gamma']
        ua_weight = current_hyperparameters['ua_weight']
        clip_grad_val = current_hyperparameters['clip_grad'] # Get clip_grad value for this trial
        patience_val = current_hyperparameters['patience'] # Get patience value for early stopping


        # 7. Initialize Model, Loss, Optimizer, Scheduler with Current Hyperparameters
        model = OpticalPropertiesEstimator(network_config)
        criterion = MSELossWeightedUA(ua_weight=ua_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Use sampled batch_size
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   # Use sampled batch_size


        # 8. Train the Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model, train_losses, val_losses, learning_rates = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=trial_epochs, device=device, clip_grad=clip_grad_val, patience=patience_val # Pass clip_grad and patience
        )

        # 9. Evaluate and Store Results
        final_val_loss = val_losses[-1] # Or could use the best validation loss from early stopping if you store it in train_model
        history.append({'hyperparameters': current_hyperparameters, 'val_loss': final_val_loss, 'train_losses': train_losses, 'val_losses': val_losses, 'learning_rates': learning_rates}) # Store more info

        print(f"Trial {trial+1} - Val Loss: {final_val_loss:.4f}")

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_hyperparameters = current_hyperparameters
            best_model = trained_model # Optionally store the best model itself


    print("\n--- Hyperparameter Search Summary ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("Best Hyperparameters:", best_hyperparameters)

    # 10. Example of Plotting Loss Curves for the best model (optional, you can modify to plot for all trials or top trials)
    print("\nLong training with best hyperparameters...")
    if best_hyperparameters:
        best_trial_history = None
        for h in history:
            if h['hyperparameters'] == best_hyperparameters:
                best_trial_history = h
                break

        if best_trial_history:
            plt.figure(figsize=(10, 6))
            plt.plot(best_trial_history['train_losses'], label='Train Loss (Best Trial)')
            plt.plot(best_trial_history['val_losses'], label='Validation Loss (Best Trial)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Curve - Best Hyperparameters')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            
    current_hyperparameters = best_hyperparameters

    network_config = {
            'cnn_filters_hist': current_hyperparameters['cnn_filters_hist'],
            'cnn_filters_irf': current_hyperparameters['cnn_filters_irf'],
            'kernel_size_cnn': current_hyperparameters['kernel_size_cnn'],
            'pool_size_cnn': current_hyperparameters['pool_size_cnn'],
            'padding_cnn': current_hyperparameters['padding_cnn'],
            'use_batchnorm_cnn': current_hyperparameters['use_batchnorm_cnn'],
            'fc_layer_sizes': current_hyperparameters['fc_layer_sizes'],
            'dropout_rate_fc': current_hyperparameters['dropout_rate_fc']
        }


    batch_size = current_hyperparameters['batch_size']
    lr = current_hyperparameters['lr']
    step_size = current_hyperparameters['step_size']*20
    gamma = current_hyperparameters['gamma']
    ua_weight = current_hyperparameters['ua_weight']
    clip_grad_val = current_hyperparameters['clip_grad'] # Get clip_grad value for this trial
    patience_val = current_hyperparameters['patience'] # Get patience value for early stopping


    # 7. Initialize Model, Loss, Optimizer, Scheduler with Current Hyperparameters
    model = OpticalPropertiesEstimator(network_config)
    criterion = MSELossWeightedUA(ua_weight=ua_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Use sampled batch_size
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   # Use sampled batch_size


    # 8. Train the Model
    train_epochs = 2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, train_losses, val_losses, learning_rates = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=train_epochs, device=device, clip_grad=clip_grad_val, patience=patience_val # Pass clip_grad and patience
    )


    print("Training finished!")


    # 6. Example Inference (after training)
    #    Load a sample histogram and IRF (e.g., from your validation set or a new measurement)
    sample_index = 3 # Example index from validation set
    sample_histogram = torch.tensor(histograms_val[sample_index], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Add batch and channel dimension
    sample_irf = torch.tensor(irfs_val[sample_index], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Add batch and channel dimension

    trained_model.eval() # Set model to evaluation mode
    with torch.no_grad():
        predicted_optical_properties = trained_model(sample_histogram, sample_irf)

    predicted_ups = predicted_optical_properties[0][0].item()
    predicted_ua = predicted_optical_properties[0][1].item()
    actual_ups = ups_val[sample_index]
    actual_ua = ua_val[sample_index]

    print(f"Predicted ups: {predicted_ups:.4f}, Actual ups: {actual_ups:.4f}")
    print(f"Predicted ua: {predicted_ua:.4f}, Actual ua: {actual_ua:.4f}")


    error_ua = []
    error_ups = []
    actual_uas = []
    actual_upss = []

    for i in range(len(histograms_val)):
        sample_histogram = torch.tensor(histograms_val[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Add batch and channel dimension
        sample_irf = torch.tensor(irfs_val[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Add batch and channel dimension

        trained_model.eval() # Set model to evaluation mode
        with torch.no_grad():
            predicted_optical_properties = trained_model(sample_histogram, sample_irf)

        predicted_ups = predicted_optical_properties[0][0].item()
        predicted_ua = predicted_optical_properties[0][1].item()
        actual_ups = ups_val[i]
        actual_ua = ua_val[i]

        actual_uas.append(actual_ua)
        actual_upss.append(actual_ups)

        error_ua.append(abs((actual_ua - predicted_ua)/actual_ua)*100)
        error_ups.append(abs((actual_ups - predicted_ups)/actual_ups)*100)
        

    print("Median error in ups: ", np.nanmedian(error_ups))
    print("Median error in ua: ", np.nanmedian(error_ua))

    # Save the trained model and hyperparameters
    model_folder = "TrainedModels"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    date_time = time.strftime("%Y%m%d%H%M%S")
    config_path = '{}/optical_propierties_estimator_config_{}.json'.format(model_folder,date_time)
    with open(config_path, 'w') as f:
        json.dump(best_hyperparameters, f, indent=4) # Save with indent for readability
    torch.save(trained_model.state_dict(), '{}/optical_properties_estimator_weights_{}.pth'.format(model_folder,date_time))