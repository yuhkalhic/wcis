from .imdb_dataset import IMDBDataset
from torch.utils.data import DataLoader


def setup_dataloaders(imdb_tokenized):
    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=4
    )

    return train_loader, val_loader, test_loader