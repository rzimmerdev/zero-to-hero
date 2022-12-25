import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import random_split
import pytorch_lightning as pl


from trainer import LitTrainer
from models import CNN


def main():
    from torch.utils.data import DataLoader
    from src.dataset import DatasetMNIST, load_mnist

    mnist = load_mnist("../downloads/mnist/")
    dataset, test_data = DatasetMNIST(*mnist["train"]), DatasetMNIST(*mnist["test"])

    train_size = round(len(dataset) * 0.8)
    validate_size = len(dataset) - train_size
    train_data, validate_data = random_split(dataset, [train_size, validate_size])

    train_dataloader = DataLoader(train_data, num_workers=6)  # My CPU has 8 cores
    validate_dataloader = DataLoader(validate_data, num_workers=2)
    test_dataloader = DataLoader(test_data, num_workers=8)  # My CPU has 8 cores

    net = CNN(input_channels=1, num_classes=10).to("cuda")
    opt = optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    max_epochs = 10
    for i in range(max_epochs):
        for idx, batch in enumerate(train_dataloader):
            x, y = batch
            x = x.to("cuda")
            y = y.to("cuda")

            y_pred = net(x).reshape(1, -1)
            loss = loss_fn(y_pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if idx % 1000 == 0:
                print(f"Loss: {loss.item()} ({idx} / {len(train_dataloader)})")

    torch.save(net, "../checkpoints/pytorch/version_1.pt")

    # grayscale channels = 1, mnist num_labels = 10
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=10, default_root_dir="../checkpoints")
    pl_net = LitTrainer(CNN(input_channels=1, num_classes=10))
    trainer.fit(pl_net, train_dataloader, validate_dataloader)
    trainer.test(model=pl_net, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
