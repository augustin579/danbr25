from pathlib import Path
from PIL import Image

def _scaler(
    image_path: Path,
    save_path: Path,
    max_size: int,
):
    try:
        img = Image.open(image_path)
        img.verify()
        img = img.convert("RGB")
        scaler = max_size / max(img.width, img.height)
        img = img.resize((
            int(img.width * scaler),
            int(img.height * scaler)
        ))
        img.save(save_path, icc_profile=None)
    except Exception as E:
        print(
            f"[ERROR] {image_path.name} is not a valid image. Make sure the directory only contains valid images."
        )
        raise


def images_scaler(
    images_dir: str | Path,
    save_dir: str | Path,
    max_size: int,
    recursive: bool=False,
):
    images_dir = Path(images_dir)
    save_dir = Path(save_dir)

    if recursive:
        for dirpath, dirnames, filenames in images_dir.walk():
            for file_name in filenames:
                image = Path(dirpath) / file_name
                if not image.is_file(): continue

                sub_dir = str(dirpath).replace(str(images_dir), "")
                save_dest = save_dir / sub_dir

                if not save_dest.exists():
                    save_dest.mkdir(parents=True, exist_ok=True)
  
                save_dest = save_dest / image.name

                _scaler(image, save_dest, max_size)
    else:
        for image in images_dir.iterdir():
            if not image.is_file(): continue

            _scaler(image, save_dir / image.name, max_size)
