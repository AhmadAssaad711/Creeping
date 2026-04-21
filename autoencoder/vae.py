import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from encoder import VariationalEncoder
from decoder import Decoder
from tqdm.auto import tqdm


# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LATENT_SPACE = 95


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.model_file = os.path.join('autoencoder/model', 'var_autoencoder.pth')
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
    
    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()

def train(model, trainloader, optimizer, beta=1.0):
    model.train()
    running = 0.0
    for x, _ in tqdm(trainloader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        x_hat = model(x)
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kl    = model.encoder.kl
        loss  = recon + beta * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(trainloader)


def test(model, testloader):
    # Set evaluation mode for encoder and decoder
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in testloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Decode data
            x_hat = model(x)
            recon = F.mse_loss(x_hat, x, reduction='mean')          # mean over pixels & batch
            kl    = model.encoder.kl                                 # mean over batch (sum over z per sample)
            loss  = recon + kl                                # beta ~ 1.0 is sensible
            val_loss += loss.item()

    return val_loss / len(testloader.dataset)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "dataset")  # not ".../autoencoder/autoencoder"
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    # Sanity: required structure for ImageFolder
    # dataset/train/class1/*.png, dataset/test/class1/*.png
    assert os.path.isdir(train_dir), f"Missing: {train_dir}"
    assert os.path.isdir(test_dir),  f"Missing: {test_dir}"

    writer = SummaryWriter("runs/auto-encoder")
    print("TB log dir:", writer.log_dir)

    train_tfms = transforms.Compose([
        transforms.Resize((80, 160)),          # <— force expected size
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((80, 160)),          # <— same size for val/test
        transforms.ToTensor(),
    ])


    # Build datasets (orig_train BEFORE split so we can inspect it)
    orig_train = datasets.ImageFolder(train_dir, transform=train_tfms)
    test_data  = datasets.ImageFolder(test_dir,  transform=test_tfms)

    print("train_dir:", train_dir)
    print("test_dir :", test_dir)
    print("orig_train size:", len(orig_train), "classes:", orig_train.classes)
    print("test size     :", len(test_data),  "classes:", test_data.classes)

    # If this prints 0 or classes == [], your train folder is empty or lacks class subfolder.
    if len(orig_train) == 0:
        raise RuntimeError(
            "Your TRAIN set is empty. Put images under dataset/train/class1 (or any class name)."
        )

    # Split train/val
    m = len(orig_train)
    n_train = int(0.8 * m)
    n_val   = m - n_train
    train_data, val_data = random_split(
        orig_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print("split -> train:", len(train_data), "val:", len(val_data))

    # DataLoaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    testloader  = torch.utils.data.DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    print("#train batches:", len(trainloader))
    print("#val   batches:", len(validloader))
    print("#test  batches:", len(testloader))

    # Model/optim
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Selected device: {device}")

    # Train
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optimizer)   # <- pass optimizer, not 'optim' name clash
        writer.add_scalar("Training Loss/epoch", train_loss, epoch+1)

        val_loss = test(model, validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)

        print(f"EPOCH {epoch+1}/{NUM_EPOCHS}\ttrain {train_loss:.3f}\tval {val_loss:.3f}")

    os.makedirs(os.path.join(base_dir, "model"), exist_ok=True)
    model.save()
    writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nTerminating...')
