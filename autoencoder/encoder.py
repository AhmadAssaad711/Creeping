import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super().__init__()

        # Prefer env override if provided, else default to encoder weights
        env_path = os.environ.get("VAE_WEIGHTS", "").strip()
        self.model_file = os.path.abspath(env_path) if env_path else os.path.join(HERE, "model", "var_encoder_model.pth")

        # --- Encoder: Conv blocks (structure unchanged) ---
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # -> (N, 32, 39, 79) for 80x160 with p=0
            nn.LeakyReLU()
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> (N, 64, 20, 40)
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # -> (N, 128, 9, 19)
            nn.LeakyReLU()
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),  # -> (N, 256, 4, 9)
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(9 * 4 * 256, 1024),
            nn.LeakyReLU()
        )

        # --- Heads: mean and log-variance (was "sigma" before) ---
        self.mu     = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

        # Keep your Normal() object (structure preserved)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

        # Expose last KL value
        self.kl = 0.0

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        # Mean and log-variance
        mu     = self.mu(x)
        logvar = self.logvar(x)

        # Reparameterization: std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = self.N.sample(mu.shape).to(mu.device)   # ~ N(0, I)
        z   = mu + std * eps

        # KL(q(z|x) || N(0,I)), summed over latent dims, mean over batch
        # D_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        self.kl = kl_per_sample.mean()

        return z

    def save(self):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        # Load checkpoint
        state = torch.load(self.model_file, map_location=device)

        # If the file contains a full VAE, strip the "encoder." prefix
        if any(k.startswith("encoder.") for k in state.keys()):
            state = {k.replace("encoder.", "", 1): v
                    for k, v in state.items() if k.startswith("encoder.")}

        # Backward-compat: old checkpoints used "sigma" head; new code uses "logvar"
        if "sigma.weight" in state:
            state["logvar.weight"] = state.pop("sigma.weight")
        if "sigma.bias" in state:
            state["logvar.bias"] = state.pop("sigma.bias")

        # Load (non-strict to tolerate harmless extras/missing)
        missing, unexpected = self.load_state_dict(state, strict=False)

        if missing or unexpected:
            print(f"[Encoder.load] Missing: {missing} | Unexpected: {unexpected}")

        self.eval()
