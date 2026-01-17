# danbr25
A simple library for downloading, plotting, and training models from danbooru2025-metadata dataset.

## Installation

```bash
# Chose one
## Install PyTorch.
## Use this command if you don't have GPU:
## Option 1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

## But if you have GPU with CUDA version 13.0, use this command:
## For another version, check the PyTorch documentation.
## Option 2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130


# Install danbr25 (Required)
pip install git+https://www.github.com/augustin579/danbr25.git
```

# download
Download images from the parquets.

## Usage

```python
from danbr25.download import download_images

# download images from the dataset
download_images(
    parquets_dir="./dataset",
    aspect_ratio={"min": 0.5, "max": 2.0},
    save_dir="./images",
    filter_ai=True
)
```

## Parameters

- `parquets_dir`: Path to directory where the parquet files stored.
- `aspect_ratio`: Aspect Ratio of allowed images. Default is {"min": 0.5, "max": 2.0}.
- `save_dir`: Path to directory where the images will be saved.
- `filter_ai`: If True, will filter out images marked as AI. Default is True.

<br/>
<br/>
<br/>

# plot

This tool helps for plotting the result of the training.

## Usage
```python
from danbr25.plot import line_plot, pie_plot, plot_cm
```

### Line Plot
```python
line_plot(
    data={
        "Line A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Line B": [5.0, 4.0, 3.0, 2.0, 1.0]
    },
    title="Line Plot",
    labels=("x-axis", "y-axis"),
    figsize=(10, 5),
    xticks=[1.0, 2.0, 3.0, 4.0, 5.0],
    yticks=[1.0, 2.0, 3.0, 4.0, 5.0],
    save_path="./line_plot.png"
)
```

#### Parameters:

- `data`: Data to be plotted. It's basically a dictionary
which the key is the name of the line and the value is
the list of values to be plotted. Each value must has the
same length.
- `title`: Title of the plot.
- `labels`: Labels of the plot.
- `figsize`: Size of the figure.
- `xticks`: Set custom x-axis ticks.
- `yticks`: Set custom y-axis ticks.
- `save_path`: Path to save the plot.


<br/>
<hr/>
<br/>

### Pie Plot
```python
pie_plot(
    data={
        "Class A": 100,
        "Class B": 200
    },
    title="Pie Plot",
    figsize=(10, 5),
    save_path="./pie_plot.png"
)
```

#### Parameters:

- `data`: Data to be plotted. It's a dictionary with
the key as the name for each sector, and the value as
the size of the sector.
- `title`: Title of the plot.
- `figsize`: Size of the figure.
- `save_path`: Path to save the plot.

<br/>
<hr/>
<br/>

### Confusion Matrix
```python
plot_cm(
    labels_data=[1, 0, 1, 1, 0],
    predictions_data=[0, 1, 1, 1, 0],
    classes_labels={
        1: "Class A",
        0: "Class B"
    },
    cm_text=[
        ["TP", "FP"],
        ["FN", "TN"]
    ],
    cm_text_color="yellow",
    cm_text_size=24,
    cmap="Blues",
    figsize=(10, 5),
    save_path="./cm.png"
)
```

#### Parameters:

- `labels_data`: List of labels.
- `predictions_data`: List of predictions.
- `classes_labels`: Dictionary of classes labels.
- `figsize`: Size of the figure.
- `cm_text`: Text to be displayed on the confusion matrix.

- `cm_text_color`: Color of the text. You can find the list of
available colors [here](https://matplotlib.org/stable/tutorials/colors/colors.html).

- `cm_text_size`: Size of the text.

- `cmap`: Colormap to be used. You can find the list of
available colormaps [here](https://matplotlib.org/stable/tutorials/colors/colormaps.html).

- `save_path`: Path to save the plot.

<br/>
<hr/>
<br/>

# training_fn

This tool is used to train models from the images dataset.

## Usage
```python
from danbr25.training_fn import manual_seed, dataset_init, train, validation, test
```

### Manual Seed
```python
manual_seed(42)
```

#### Parameters:

- `seed`: An arbitrary integer to be used as seed.

<br/>
<br/>
<br/>

### Dataset Initialization
The ```dataset_init``` function is used to initialize the dataset.

```python
from torchvision.transforms import v2
from torchvision.models import ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2

train_transforms = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(
        weights.meta["std"],
        weights.meta["mean"]
    )
])

validation_transforms = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.Normalize(
        weights.meta["std"],
        weights.meta["mean"]
    )
])

dataset_init(
    dataset_dir="./dataset",
    batch_size=32,
    num_workers=4,
    train_transforms=train_transforms,
    val_transforms=validation_transforms
)
```


#### Parameters:

- `dataset_dir`: Path to directory where the dataset is stored.
- `batch_size`: Batch size.
- `num_workers`: Number of workers.
- `train_transforms`: Training transforms.
- `validation_transforms`: Validation transforms.

#### Return:

It returns a dictionary of data loaders. The dictionary has the following keys:

- `train`: Training data loader.
- `validation`: Validation data loader.
- `test`: Test data loader.


#### Note:

The ```dataset_dir``` must be a path to directory where the dataset is stored in the following structure:

```
dataset_dir/
    ├── class_1/
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── class_2/
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    └── ...
```

<br/>
<hr/>
<br/>

### Training
```python
from torchvision.models import resnet50
from torch.optim import Adam

model = resnet50(weights="IMAGENET1K_V2")
loader = dataset_init(
    dataset_dir="./dataset",
    batch_size=32,
    num_workers=4,
    train_transforms=train_transforms,
    val_transforms=validation_transforms
)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
scaler = torch.cuda.amp.GradScaler()

train(
    model=model,
    loaders=loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    scaler=scaler
)
```

#### Parameters:

- `model`: Model to be trained.
- `loaders`: Data loaders.
- `optimizer`: Optimizer.
- `loss_fn`: Loss function.
- `device`: Device to be used.
- `scaler`: Scaler.

#### Return:

It returns a training loss.

<br/>
<hr/>
<br/>

### Validation
```python

classes = {
    "Class A": 0,
    "Class B": 1
}

validation(
    model=model,
    loader=loader,
    loss_fn=loss_fn,
    device=device,
    classes=classes
)
```

#### Parameters:

- `model`: Model to be validated.
- `loader`: Data loader.
- `loss_fn`: Loss function.
- `device`: Device to be used.
- `classes`: Classes labels.

#### Return:

It returns a tuple of validation loss, precision, recall, and f1-score.

<br/>
<hr/>
<br/>

### Test
```python
test(
    model=model,
    loader=loader,
    loss_fn=loss_fn,
    device=device,
    classes=classes
)
```

#### Parameters:

- `model`: Model to be tested.
- `loader`: Data loader.
- `loss_fn`: Loss function.
- `device`: Device to be used.
- `classes`: Classes labels.

#### Return:

It returns a dictionary of metrics from ```sklearn.metrics.classification_report```, supplemented with the following extra keys:

- `all_predictions`: All predictions.
- `all_labels`: All labels.

Excluded keys: ```weighted avg```

#### Note:

You can find the list of available metrics [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).