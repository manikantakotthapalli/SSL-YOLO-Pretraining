import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cspdarknet import CSPDarknetBackbone
import os

# Projection head
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# SimCLR model
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(encoder.out_dim, out_dim=projection_dim)

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)

# NT-Xent Loss
def nt_xent_loss(out_1, out_2, temperature=0.5):
    batch_size = out_1.size(0)
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature

    labels = torch.arange(batch_size).cuda()
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(batch_size * 2, dtype=torch.bool).cuda()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    positives = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)], dim=0)
    nominator = torch.exp(positives)
    denominator = torch.sum(torch.exp(sim_matrix), dim=1)
    loss = -torch.log(nominator / denominator).mean()
    return loss

# Data augmentations for SimCLR
simclr_augment = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=None)
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.dataset)

# Training loop
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    data_path = './dataset/coco_unlabeled/'  # Folder must contain a dummy class folder
    dataset = SimCLRDataset(root_dir=data_path, transform=simclr_augment)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # Model
    encoder = CSPDarknetBackbone()
    model = SimCLR(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for x1, x2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

        if epoch %10 ==0:
            torch.save(model.encoder.state_dict(), 'ssl_cspdarknet_epoch50.pth')

    # Save backbone encoder weights
    torch.save(model.encoder.state_dict(), 'ssl_cspdarknet.pth')
