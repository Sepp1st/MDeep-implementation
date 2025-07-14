import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkBinary(nn.Module):
    def __init__(self, args, num_features):
        super(NetworkBinary, self).__init__()
        filter_num = args.filter_num
        window_size = args.window_size
        pool_strides = args.strides
        forward_size = args.forward_size
        dropout_rate = args.dropout_rate

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=filter_num[0],
            kernel_size=window_size[0],
            stride=1,
            padding=(window_size[0] - 1) // 2
        )
        self.bn1 = nn.BatchNorm1d(filter_num[0])
        self.pool1 = nn.MaxPool1d(kernel_size=pool_strides[0], stride=pool_strides[0])
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv1d(
            in_channels=filter_num[0],
            out_channels=filter_num[1],
            kernel_size=window_size[1],
            stride=1,
            padding=(window_size[1] - 1) // 2
        )
        self.bn2 = nn.BatchNorm1d(filter_num[1])
        self.pool2 = nn.MaxPool1d(kernel_size=pool_strides[1], stride=pool_strides[1])
        self.dropout2 = nn.Dropout(p=dropout_rate)

        conv_output_size = self._get_conv_output_size(num_features)

        # --- Fully-Connected Layers ---
        self.fc1 = nn.Linear(in_features=conv_output_size, out_features=forward_size)
        self.bn3 = nn.BatchNorm1d(forward_size)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        self.fc_out = nn.Linear(in_features=forward_size, out_features=2)

        self._initialize_conv_layers()
        self._initialize_linear_layers()

    def _get_conv_output_size(self, num_features):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, num_features)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return x.numel()

    def _initialize_conv_layers(self):
        # Your original initialization for convolutional layers
        print("Initializing convolutional layers with Xavier Uniform...")
        for m in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def _initialize_linear_layers(self):
        # Your original initialization for linear layers
        print("Initializing linear layers with Truncated Normal...")
        for m in [self.fc1, self.fc_out]:
            nn.init.trunc_normal_(m.weight.data, std=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        # The forward pass logic

        # Conv Path
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # FC Path
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)

        logits = self.fc_out(x)
        return logits