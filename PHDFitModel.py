import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.signal import savgol_filter



def move_array_ypos(arr, xpos):
    '''
    Move array to the right by xpos
    '''
    arr = np.roll(arr, xpos)
    if xpos > 0:
        arr[:xpos] = 0
    elif xpos < 0:
        arr[xpos:] = 0
    return arr
    

def normalize_dtof(signal, cut_threshold=0.01):
        """Normalizes a 1D signal to have an area (sum) of 1.

        Args:
            signal (torch.Tensor): A tensor representing a 1D signal, shape (..., length).
                                Assumes the last dimension is the sequence length.
            cut_threshold (float): A threshold to cut off low values in the signal.

        Returns:
            torch.Tensor: Normalized signal, shape (..., length).
                        Returns the original signal if its sum is zero to avoid division by zero.
        """
        normalized_signal = signal/np.max(savgol_filter(signal, 50, 3)) # Normalize by the maximum value
        cut_threshold = cut_threshold * np.max(normalized_signal) # Calculate the cut threshold
        normalized_signal[normalized_signal < cut_threshold] = 0 # Set values below the threshold to zero
        
        return normalized_signal

class TurbidMediaDataset(Dataset):
    def __init__(self, histograms, irfs, ups, ua):
        self.histograms = histograms
        self.irfs = irfs
        self.ups = ups
        self.ua = ua

    def __len__(self):
        return len(self.histograms)

    def __getitem__(self, idx):
        histogram = normalize_dtof(self.histograms[idx]) # Normalize histogram
        histogram = torch.tensor(histogram, dtype=torch.float32).unsqueeze(0) # Add channel dimension
        irf = normalize_dtof(self.irfs[idx]) # Normalize IRF
        irf = torch.tensor(irf, dtype=torch.float32).unsqueeze(0) # Add channel dimension
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
        with torch.no_grad():
            hist_cnn_out_example = self.hist_cnn(example_input) # Shape: (Batch, Channels_Hist, SeqLen')
            irf_cnn_out_example = self.irf_cnn(example_input)   # Shape: (Batch, Channels_IRF, SeqLen')

        cnn_output_channels_hist = hist_cnn_out_example.shape[1]
        cnn_output_channels_irf = irf_cnn_out_example.shape[1]
        self.cnn_output_seq_len = hist_cnn_out_example.shape[2] # Assuming hist and irf branches have same pooling/stride
        
        #hist_cnn_output_size = self._get_cnn_output_size(self.hist_cnn, example_input)
        #irf_cnn_output_size = self._get_cnn_output_size(self.irf_cnn, example_input)
        #combined_cnn_output_size = hist_cnn_output_size + irf_cnn_output_size
        
        # --- RNN Layer ---
        self.use_rnn = config.get('use_rnn', False) # Control flag
        if self.use_rnn:
            rnn_input_size = cnn_output_channels_hist + cnn_output_channels_irf
            self.rnn_hidden_dim = config.get('rnn_hidden_dim', 128)
            rnn_layers = config.get('rnn_layers', 1)
            self.rnn_bidirectional = config.get('rnn_bidirectional', True)
            rnn_dropout = config.get('rnn_dropout', 0.0) # Dropout between RNN layers if rnn_layers > 1

            # Using LSTM, GRU is an alternative: nn.GRU(...)
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=self.rnn_hidden_dim,
                num_layers=rnn_layers,
                batch_first=True, # Input/Output shape: (Batch, Seq, Features)
                dropout=rnn_dropout if rnn_layers > 1 else 0,
                bidirectional=self.rnn_bidirectional
            )
            
            # Output dimension of the RNN (for the final FC layer)
            rnn_output_dim = self.rnn_hidden_dim * (2 if self.rnn_bidirectional else 1)
            in_features = rnn_output_dim # FC input size is RNN output size

        else:
            # If not using RNN, fall back to concatenating flattened CNN features (original approach)
            # Note: This part needs adjustment if you *require* RNN/Attention
            combined_cnn_output_size = (cnn_output_channels_hist + cnn_output_channels_irf) * self.cnn_output_seq_len
            rnn_output_dim = combined_cnn_output_size # Treat flattened CNN output as the feature vector
            in_features = rnn_output_dim

        # --- Attention Layer (Optional, applied to RNN sequence output) ---
        self.use_attention = config.get('use_attention', True) # Control flag
        if self.use_rnn and self.use_attention:
            self.attention_dim = rnn_output_dim
            # Example: Simple additive attention (Bahdanau-style variant)
            # We compute scores based on the RNN output sequence
            self.attention_W = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
            self.attention_v = nn.Linear(self.attention_dim, 1, bias=False)
            # The input feature size to the FC layers will be self.attention_dim
            in_features = self.attention_dim
        #elif self.use_rnn:
        #     # If using RNN but not Attention, use the RNN's last hidden state or last output step
        #     # Using last output step is simpler here
        #     fc_input_features = rnn_output_dim
        #else:
        #     # If not using RNN, input features are from flattened CNNs
        #     in_features = rnn_output_dim # Which is combined_cnn_output_size in this case


        # Fully connected layers with BatchNorm
        fc_layers = []
        fc_layer_sizes = config.get('fc_layer_sizes', [128, 64]) # Default FC layer sizes
        dropout_rate_fc = config.get('dropout_rate_fc', 0.2) # Configurable dropout
        #in_features = combined_cnn_output_size

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
        hist_cnn_output = self.hist_cnn(histogram) # Shape: (Batch, Channels_Hist, SeqLen')
        irf_cnn_output = self.irf_cnn(irf)       # Shape: (Batch, Channels_IRF, SeqLen')

        if self.use_rnn:
            # --- RNN Processing ---
            # Concatenate along the channel dimension (dim=1)
            combined_cnn_output = torch.cat((hist_cnn_output, irf_cnn_output), dim=1) # Shape: (Batch, Channels_Hist + Channels_IRF, SeqLen')

            # Permute for RNN (batch_first=True): (Batch, SeqLen', Features)
            # Features = Channels_Hist + Channels_IRF
            rnn_input = combined_cnn_output.permute(0, 2, 1) # Shape: (Batch, SeqLen', Channels_Hist + Channels_IRF)

            # Pass through RNN
            # rnn_output_seq shape: (Batch, SeqLen', NumDirections * HiddenDim)
            # h_n shape: (NumLayers * NumDirections, Batch, HiddenDim)
            rnn_output_seq, (h_n, c_n) = self.rnn(rnn_input)

            if self.use_attention:
                # --- Attention Mechanism ---
                # Simple Additive Attention Example:
                # Score each RNN output step
                # energi = tanh(W * h_t)
                energy = torch.tanh(self.attention_W(rnn_output_seq)) # (Batch, SeqLen', attention_dim)
                # attention_scores = v * energi -> (Batch, SeqLen', 1)
                attention_scores = self.attention_v(energy).squeeze(2) # (Batch, SeqLen')
                # Normalize scores to get weights
                attention_weights = F.softmax(attention_scores, dim=1) # (Batch, SeqLen')
                # Weighted sum of RNN outputs: context_vector = sum(weights * rnn_output)
                # Unsqueeze weights: (Batch, 1, SeqLen')
                # rnn_output_seq: (Batch, SeqLen', rnn_output_dim)
                context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_output_seq).squeeze(1) # (Batch, rnn_output_dim)

                features_for_fc = context_vector # Use attention context vector

            else: # Use RNN but no attention
                # Option 1: Use the output of the last time step
                # features_for_fc = rnn_output_seq[:, -1, :] # (Batch, rnn_output_dim)

                # Option 2: Use the final hidden state (needs care with bidirectional)
                # If bidirectional, h_n contains forward and backward states concatenated.
                # Example: Get last layer's hidden state
                if self.rnn_bidirectional:
                     # Concatenate final forward and backward hidden states of the last layer
                     # h_n shape: (NumLayers * NumDirections, Batch, HiddenDim)
                     # Last layer fwd: h_n[-2,:,:] ; Last layer bwd: h_n[-1,:,:]
                     features_for_fc = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # (Batch, 2 * HiddenDim)
                else:
                     # Just the final hidden state of the last layer
                     features_for_fc = h_n[-1,:,:] # (Batch, HiddenDim)
                # Using last output step might be simpler:
                features_for_fc = rnn_output_seq[:, -1, :]


        # Use the output features of the last time step for the FC layers
        # This is a common approach for sequence classification/regression
        #    features_for_fc = rnn_output_seq[:, -1, :] # Shape: (Batch, NumDirections * HiddenDim)

        else:
            # If not using RNN, flatten and combine the CNN features
            hist_features_flat = hist_cnn_output.view(hist_cnn_output.size(0), -1)
            irf_features_flat = irf_cnn_output.view(irf_cnn_output.size(0), -1)
            features_for_fc = torch.cat((hist_features_flat, irf_features_flat), dim=1) # Shape: (Batch, Flattened_Hist + Flattened_IRF)

        output = self.fc(features_for_fc)
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


class MAELossWeightedUA(nn.Module):
    def __init__(self, ua_weight=1.0):
        super().__init__()
        self.ua_weight = ua_weight
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, predictions, targets):
        #log_targets = torch.log(torch.clamp(targets, min=1e-9))
        abs_error = self.mae_loss(predictions, targets)
        ups_error = abs_error[:, 0]
        ua_error = abs_error[:, 1]
        weighted_ua_error = self.ua_weight * ua_error
        loss = torch.mean(ups_error + weighted_ua_error)
        return loss

class HuberLossWeightedUA(nn.Module):
    def __init__(self, delta=1.0, ua_weight=1.0):
        super().__init__()
        self.delta = delta
        self.ua_weight = ua_weight
        self.huber_loss = nn.HuberLoss(delta=self.delta, reduction='none')
        
    def forward(self, predictions, targets):
        #log_targets = torch.log(torch.clamp(targets, min=1e-9))
        abs_error = self.huber_loss(predictions, targets)
        ups_error = abs_error[:, 0]
        ua_error = abs_error[:, 1]
        weighted_ua_error = self.ua_weight * ua_error
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

                #histograms_normalized = normalize_dtof(histograms)
                #irfs_normalized = normalize_dtof(irfs)

                outputs = model(histograms, irfs) # Use normalized inputs
                val_loss = criterion(outputs, torch.stack((ups, ua), dim=1))
                val_running_loss += val_loss.item()


        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        #scheduler.step()
        scheduler.step(epoch_val_loss) # Metric for ReduceLROnPlateau
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

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