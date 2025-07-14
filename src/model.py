import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkBinary(nn.Module):
    def __init__(self, args):
        super(NetworkBinary, self).__init__()

        num_filter = args.kernel_size
        window_size = args.window_size
        stride_size = args.strides

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=num_filter[0],
            kernel_size=window_size[0],
            stride=stride_size[0],
            padding='same'
        )
        self.bn1 = nn.BatchNorm1d(num_filter[0])
        self.dropout1 = nn.Dropout1d(p=args.dropout_rate)

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv1d(
            in_channels=num_filter[0],
            out_channels=num_filter[1],
            kernel_size=window_size[1],
            stride=stride_size[1],
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(num_filter[1])
        self.dropout2 = nn.Dropout1d(p=args.dropout_rate)

        # Initialize convolutional layers
        self._initialize_conv_layers()

        # --- Fully-Connected Layers ---
        self.fc1 = nn.Linear(in_features=0, out_features=64)  # Placeholder in_features, defined dynamically
        self.bn3 = nn.BatchNorm1d(64)

        # Output layer for binary classification (2 classes)
        self.fc_out = nn.Linear(in_features=64, out_features=2)
        self._initialize_linear_layer(self.fc_out)

    def _initialize_conv_layers(self):
        for m in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def _initialize_linear_layer(self, m):
        nn.init.trunc_normal_(m.weight.data, std=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        # x starts as (batch, channels, length)

        # Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Flatten the output for the FC layers
        x = torch.flatten(x, start_dim=1)

        # Dynamically update the in_features for the first FC layer if it's the first pass
        if self.fc1.in_features == 0:
            self.fc1.in_features = x.shape[1]
            # Must move the newly-sized weight to the correct device
            self.fc1.weight = nn.Parameter(self.fc1.weight.new_empty(self.fc1.out_features, self.fc1.in_features).to(x.device))
            self.fc1.bias = nn.Parameter(self.fc1.bias.new_empty(self.fc1.out_features).to(x.device))
            self._initialize_linear_layer(self.fc1)

        # FC Block 1
        x = self.fc1(x)
        x = self.bn3(x)
        x = torch.tanh(x)

        # Output Layer
        logits = self.fc_out(x)

        return logits