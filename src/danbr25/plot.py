from pathlib import Path
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix

custom_style = {
    "axes.axisbelow": True,
    "axes.facecolor": "#EAE5FA",
    "axes.grid": True,
    "axes.labelsize": "xx-large",
    "axes.labelpad": 20.0,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlesize": "xx-large",
    "axes.titlepad": 20.0,
    "axes.titleweight": "bold",
    "font.family": "monospace",
    "font.size": 20,
    "grid.color": "#FEFEFE",
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    "legend.fontsize": "x-large",
    "lines.linewidth": 2.5,
    "xtick.color": "none",
    "xtick.labelcolor": "black",
    "ytick.color": "none",
    "ytick.labelcolor": "black"
}


def line_plot(
    data: dict[str, list[float]],
    title: str,
    labels: tuple[str, str],
    figsize: tuple[int, int],
    xticks: list[int] | None = None,
    yticks: list[int] | None = None,
    save_path: str | Path | None = None,
):
    pyplot.style.use(custom_style)

    fig, ax = pyplot.subplots(figsize=figsize, dpi=300, layout="constrained")

    ax.set_title(title)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    if xticks: ax.set_xticks(xticks)
    if yticks: ax.set_yticks(yticks)

    for label, axis in data.items():
        ax.plot(xticks, axis, label=label)

    ax.legend()

    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else: pyplot.show()


def pie_plot(
    data: dict[str, float],
    title: str,
    figsize: tuple[int, int],
    save_path: str | Path | None = None,
):
    pyplot.style.use(custom_style)

    fig, ax = pyplot.subplots(figsize=figsize, dpi=300, layout="constrained")

    if title: ax.set_title(title)

    labels, list_data = zip(*data.items())

    ax.pie(list_data, labels=labels, autopct="%1.1f%%")

    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else: pyplot.show()


def plot_cm(
    labels_data: list[int],
    predictions_data: list[int],
    classes_labels: dict[int, str],
    figsize: tuple[int, int],
    cm_text: list[list[str]] | None = None,
    cm_text_color: str = "yellow",
    cm_text_size: int=24,
    cmap: str = "Blues",
    save_path: str | Path | None = None
):
    pyplot.style.use(custom_style)

    cm = confusion_matrix(labels_data, predictions_data)
    classes_int, classes_str = [
        (list(_int), list(_str))
        for _int, _str in
        classes_labels.items()
    ]

    fig, ax = pyplot.subplots(
        figsize=figsize, dpi=300, layout="constrained"
    )

    ax.set_title("Confusion Matrix")
    ax.set_xticks(classes_int, classes_str)
    ax.set_yticks(classes_int, classes_str)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")

    for spine in ax.spines.values():
        spine.set_visible(True)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm_text[i][j]}\n{cm[i][j]}" if cm_text else f"{cm[i][j]}",
                ha="center", va="center", color=cm_text_color,
                size=cm_text_size
            )

    ax.imshow(cm, cmap=cmap)
    ax.grid(False)

    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else:
        pyplot.show()

