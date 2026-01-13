import torch
from torch import nn
from torchvision.models import resnet18
import math


def save_hook(module, input, output):
    setattr(module, 'output', output)




class StackedSinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding broadcast over K.
    Input: t [B, K, 1] -> emb [B, K, E]; E must be even.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        assert emb_dim % 2 == 0, "time embedding dim should be even"
        self.emb_dim = int(emb_dim)
        half_dim = emb_dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim) * -emb_scale)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,K,1]
        angles = t * self.freqs.view(1, 1, -1).to(t.dtype)  # [B,K,half]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B,K,E]


def _flatten_position(x: torch.Tensor) -> torch.Tensor:
    """Treat any input as a 'position' vector: [B, ...] -> [B, D]."""
    if x.dim() < 2:
        raise ValueError(f"Expected tensor with batch dim, got {tuple(x.shape)}")
    return x.view(x.shape[0], -1)

class Reconstructor(nn.Module):
    def __init__(self, reconstructor_type, dim_index, dim_time, channels=3, pool_size=1):
        super(Reconstructor, self).__init__()
        self.reconstructor_type = reconstructor_type
        self.dim_index = dim_index
        self.dim_time = dim_time
        self.channels = channels
        self.pool_size = pool_size
        if self.pool_size > 1:
            self.avg_pool = nn.AvgPool2d(kernel_size=(self.pool_size, self.pool_size), stride=self.pool_size)

        # === Mock (toy) ===
        if self.reconstructor_type == 'LazyLinear':
            # Inputs: positions of two points (x1, x2).
            # We compute squared distance d^2 and embed it sinusoidally, then run a 3-layer MLP where the embedding is
            # appended to each layer input.
            self.embed_dim = 64  # must be even
            self.hidden_dim = 256

            # d^2: [B,1] -> emb: [B,E]
            self.dist_embed = StackedSinusoidalPositionEmbeddings(self.embed_dim)

            # 3 linear layers with embedding appended at each layer.
            # Use LazyLinear so this works even if x1/x2 are images (we flatten them).
            self.fc1 = nn.Linear(self.channels * 2 + self.embed_dim, self.hidden_dim)
            self.bn1 = nn.BatchNorm1d(self.hidden_dim)

            self.fc2 = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
            self.bn2 = nn.BatchNorm1d(self.hidden_dim)

            self.fc3 = nn.Linear(self.hidden_dim + self.embed_dim, self.hidden_dim)
            self.bn3 = nn.BatchNorm1d(self.hidden_dim)

            # SiLU tends to keep gradients healthier than ReLU in small MLPs; BN further stabilizes scale.
            self.act = nn.SiLU(inplace=True)

            # Heads
            self.path_indices = nn.Linear(self.hidden_dim + self.embed_dim, self.dim_index+self.dim_time)
            self.shift_magnitudes = nn.Linear(self.hidden_dim + self.embed_dim, 2)

            # Init scheme (simple/standard, consistent with typical torch usage)
            for m in [self.fc1, self.fc2, self.fc3, self.path_indices, self.shift_magnitudes]:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # === LeNet ===
        elif self.reconstructor_type == 'LeNet':
            # Define LeNet backbone for feature extraction
            self.lenet_width = 2
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(self.channels * 2, 3 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(3 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(3 * self.lenet_width, 8 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(8 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(8 * self.lenet_width, 60 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(60 * self.lenet_width),
                nn.ReLU()
            )

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, self.dim_index)
            )


            # Define regression head (for predicting shift magnitudes)
            self.shift_magnitudes = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, 2)
            )

        # === ResNet ===
        elif self.reconstructor_type == 'ResNet':
            # Define ResNet18 backbone for feature extraction
            self.features_extractor = resnet18(pretrained=False)
            # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
            self.features_extractor.conv1 = nn.Conv2d(in_channels=6,
                                                      out_channels=64,
                                                      kernel_size=(7, 7),
                                                      stride=(2, 2),
                                                      padding=(3, 3), bias=False)
            nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.features = self.features_extractor.avgpool
            self.features.register_forward_hook(save_hook)

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Linear(512, self.dim_index)

            self.shift_magnitudes = nn.Linear(512, 2)

    def forward(self, x1, x2):
        if self.reconstructor_type == "LazyLinear":
            x1 = _flatten_position(x1)
            x2 = _flatten_position(x2)
            d2 = (x1 - x2).pow(2).sum(dim=-1, keepdim=True)  # [B,1]
            emb = self.dist_embed(d2[:, None, :]).squeeze(1)  # [B,E]

            h = torch.cat([x1, x2, emb], dim=-1)
            h = self.act(self.bn1(self.fc1(h)))
            h = self.act(self.bn2(self.fc2(torch.cat([h, emb], dim=-1))))
            h = self.act(self.bn3(self.fc3(torch.cat([h, emb], dim=-1))))

            h_e = torch.cat([h, emb], dim=-1)
            return self.path_indices(h_e).view(x1.shape[0], -1), self.shift_magnitudes(h_e).view(x1.shape[0], -1)

        if self.pool_size > 1:
            x1 = self.avg_pool(x1)
            x2 = self.avg_pool(x2)
        if self.reconstructor_type == 'LeNet':
            features = self.feature_extractor(torch.cat([x1, x2], dim=1))
            features = features.mean(dim=[-1, -2]).view(x1.shape[0], -1)
            return self.path_indices(features).view(x1.shape[0], -1), self.shift_magnitudes(features).view(x1.shape[0], -1)
        elif self.reconstructor_type == 'ResNet':
            self.features_extractor(torch.cat([x1, x2], dim=1))
            features = self.features.output.view([x1.shape[0], -1])
            return self.path_indices(features).view(x1.shape[0], -1), self.shift_magnitudes(features).view(x1.shape[0], -1)
        
        else:
            raise ValueError(f"Unknown reconstructor_type: {self.reconstructor_type}")

