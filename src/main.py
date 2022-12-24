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

    # grayscale channels = 1, mnist num_labels = 10
    net = CNN(input_channels=1, num_classes=10)

    pl_net = LitTrainer(net, nn.CrossEntropyLoss(), optim.Adam(net.parameters()))
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, default_root_dir="../checkpoints")

    trainer.fit(model=pl_net, train_dataloaders=train_dataloader, val_dataloaders=validate_dataloader)
    trainer.test(model=pl_net, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
