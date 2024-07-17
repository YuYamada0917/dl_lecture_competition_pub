import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)


class WavenetBlock(nn.Module):
    def __init__(self, n_channels, n_layers, kernel_size, dilation_rate):
        super(WavenetBlock, self).__init__()
        self.dilated_convs = nn.ModuleList([nn.Conv1d(n_channels, n_channels, kernel_size, 
                                           dilation=dilation_rate**i, padding='same') 
                                           for i in range(n_layers)])
        self.res_convs = nn.ModuleList([nn.Conv1d(n_channels, n_channels, 1) 
                                        for _ in range(n_layers)])
        self.skip_convs = nn.ModuleList([nn.Conv1d(n_channels, n_channels, 1) 
                                         for _ in range(n_layers)])

    def forward(self, x):
        skip = 0
        for i, (dilated_conv, res_conv, skip_conv) in enumerate(zip(self.dilated_convs, self.res_convs, self.skip_convs)):
            residual = x
            x = F.relu(dilated_conv(x))
            x = res_conv(x)
            x = x + residual
            skip = skip + skip_conv(x)
        return x, skip

class WavenetClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels, n_layers=10, n_blocks=3, kernel_size=3):
        super(WavenetClassifier, self).__init__()
        self.start_conv = nn.Conv1d(in_channels, 64, 1)
        self.blocks = nn.ModuleList([WavenetBlock(64, n_layers, kernel_size, 2) for _ in range(n_blocks)])
        self.end_conv = nn.Conv1d(64, 128, 1)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip
        x = F.relu(skip_connections)
        x = F.relu(self.end_conv(x))
        x = self.global_pooling(x).squeeze(-1)
        x = self.fc(x)
        return x
    
class WavenetBlock(nn.Module):
    def __init__(self, n_channels, n_layers, kernel_size, dilation_rate):
        super(WavenetBlock, self).__init__()
        self.dilated_convs = nn.ModuleList([nn.Conv1d(n_channels, n_channels, kernel_size, 
                                           dilation=dilation_rate**i, padding='same') 
                                           for i in range(n_layers)])
        self.res_convs = nn.ModuleList([nn.Conv1d(n_channels, n_channels, 1) 
                                        for _ in range(n_layers)])
        self.skip_convs = nn.ModuleList([nn.Conv1d(n_channels, n_channels, 1) 
                                         for _ in range(n_layers)])

    def forward(self, x):
        skip = 0
        for i, (dilated_conv, res_conv, skip_conv) in enumerate(zip(self.dilated_convs, self.res_convs, self.skip_convs)):
            residual = x
            x = F.relu(dilated_conv(x))
            x = res_conv(x)
            x = x + residual
            skip = skip + skip_conv(x)
        return x, skip

class WavenetClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels, n_layers=10, n_blocks=3, kernel_size=3):
        super(WavenetClassifier, self).__init__()
        self.start_conv = nn.Conv1d(in_channels, 64, 1)
        self.blocks = nn.ModuleList([WavenetBlock(64, n_layers, kernel_size, 2) for _ in range(n_blocks)])
        self.end_conv = nn.Conv1d(64, 128, 1)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip
        x = F.relu(skip_connections)
        x = F.relu(self.end_conv(x))
        x = self.global_pooling(x).squeeze(-1)
        x = self.fc(x)
        return x
    
class RegularizedWavenetClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels, n_layers=10, n_blocks=5, kernel_size=5, dropout=0.3):
        super(RegularizedWavenetClassifier, self).__init__()
        self.wavenet = WavenetClassifier(num_classes, seq_len, in_channels, n_layers, n_blocks, kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.wavenet.start_conv(x)
        skip_connections = 0
        for block in self.wavenet.blocks:
            x, skip = block(x)
            x = self.dropout(x)
            skip_connections = skip_connections + skip
        x = F.relu(skip_connections)
        x = F.relu(self.wavenet.end_conv(x))
        x = self.wavenet.global_pooling(x).squeeze(-1)
        x = self.dropout(x)
        x = self.wavenet.fc(x)
        return x