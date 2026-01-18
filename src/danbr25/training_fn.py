import torch
from torch.utils import data
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

class TransformedSubset(data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_init(dataset_dir, batch_size, num_workers, train_transform, validation_transform):
    pin_memory = torch.cuda.is_available()
    dataset = ImageFolder(dataset_dir)
    dataset_len = len(dataset)
    train_len = int(0.8 * dataset_len)
    validation_len = int(0.1 * dataset_len)
    test_len = dataset_len - train_len - validation_len

    train_data, validation_data, test_data = data.random_split(
        dataset,
        [train_len, validation_len, test_len]
    )
    train_data = TransformedSubset(train_data, train_transform)
    validation_data = TransformedSubset(validation_data, validation_transform)
    test_data = TransformedSubset(test_data, validation_transform)

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,num_workers=num_workers,
        persistent_workers=True, pin_memory=pin_memory
    )
    validation_loader = data.DataLoader(validation_data, batch_size=batch_size)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    loaders = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader
    }

    return loaders


def train(model, loader, optimizer, loss_fn, device, scaler):
    use_amp = (device.type == "cuda" and scaler is not None)
    model.train()
    total_loss = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

    train_loss = total_loss/total

    return train_loss


@torch.no_grad()
def validation(model, loader, loss_fn, device, classes):
    model.eval()
    total = 0
    total_loss = 0
    all_labels = []
    all_predictions = []

    tp = fp = fn = tn = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

        all_labels.extend(labels.cpu().squeeze().tolist())
        all_predictions.extend(outputs.cpu().argmax(dim=1).squeeze().tolist())


    validation_loss = total_loss/total

    report = classification_report(
        all_labels, all_predictions,
        labels=list(classes.values()),
        target_names=list(classes.keys()),
        output_dict=True
    )

    return (
        validation_loss,
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"]
    )


@torch.no_grad()
def test(model, loader, loss_fn, device, classes):
    model.eval()
    total_loss = 0
    total = 0
    all_predictions = []
    all_labels = []

    tp = fp = fn = tn = 0


    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

        all_labels.extend(labels.cpu().squeeze().tolist())
        all_predictions.extend(predictions.cpu().squeeze().tolist())

    test_loss = total_loss/total

    report = classification_report(
        all_labels, all_predictions,
        labels=list(classes.values()),
        target_names=list(classes.keys()),
        output_dict=True
    )

    report["test_loss"] = test_loss
    report["all_predictions"] = all_predictions
    report["all_labels"] = all_labels

    del report["weighted avg"]

    return report


