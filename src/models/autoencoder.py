import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Baseline conv AE from the project guide.

    3x256x256 -> 32 -> 64 -> latent channels at 32x32 -> back to 3x256x256 (sigmoid, [0,1] inputs).
    """

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3, 32), nn.MaxPool2d(2),
            conv_block(32, 64), nn.MaxPool2d(2),
            conv_block(64, latent_channels), nn.MaxPool2d(2),
        )

        def up():
            return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.decoder = nn.Sequential(
            up(), conv_block(latent_channels, 64),
            up(), conv_block(64, 32),
            up(), conv_block(32, 16),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_model(cfg: dict) -> ConvAutoencoder:
    return ConvAutoencoder(latent_channels=cfg.get("latent_channels", 128))
