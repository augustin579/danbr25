from pathlib import Path
from random import uniform
from re import sub
from time import sleep
from typing import Literal
import urllib3
from pandas import read_parquet



def _download(
    url: str,
    save_file_path: Path,
    max_retry: int = 3,
    timeout: float = 15,
    backoff_factor: float = 0.5
):
    attemp = 0

    while attemp < max_retry:
        try:
            res = urllib3.request("GET", url, preload_content=False, timeout=timeout)

            if res.status == 429: raise urllib3.exceptions.HTTPError()

            with open(save_file_path, mode="wb") as IMG:
                for chunk in res.stream(512):
                    IMG.write(chunk)
                    IMG.flush

            res.release_conn()

            return True
        except urllib3.exceptions.HTTPError as E:
            if res.status == 429:
                print(f"[WARNING] Rate Limited for {url}")
                sleep_time = backoff_factor * 2**attemp + uniform(0, 1)
                print(f"Waiting for {sleep_time:.3f}. Retry: {attemp}/{max_retry}")
                sleep(sleep_time)
            else:
                print(f"[ERROR] HTTPError with Status Code: {res.status} for {url}")
                print(f"{E}. Skipping...")
                return False
        except urllib3.exceptions.ResponseError as E:
            print(f"[WARNING] Network Error for {url}")
            sleep_time = backoff_factor * 2**attemp + uniform(0, 1)
            print(f"Reconnect. Retry: {attemp}/{max_retry}")
            sleep(sleep_time)
        except Exception as E:
            print(f"[ERROR] Unexpected Exception occured for {url}")
            print(f"{E}. Skipping...")
            return False

        attemp += 1



def download_images(
    parquets_dir: str,
    aspect_ratio: dict[Literal["min", "max"], float],
    save_dir: str,
    filter_ai: bool=True,
    max_retry: int = 3,
    timeout: float = 15,
    backoff_factor: float = 0.5
):
    use_column = ['id', 'file_url', 'file_ext', 'tag_string', 'image_height', 'image_width']
    for parquet_file in Path(parquets_dir).iterdir():
        if not ".parquet" in parquet_file.name: continue

        dataset = read_parquet(parquet_file, engine="pyarrow", columns=use_column)

        mask = dataset["file_ext"].isin(["jpg", "png"])

        dataset = dataset[mask]

        if filter_ai:
            mask = dataset["tag_string"].str.contains("ai-generated|ai-assisted")
        else:
            mask = ~(dataset["tag_string"].str.contains("ai-generated|ai-assisted"))

        dataset = dataset[mask]


        ar_data = dataset['image_height'] / dataset['image_width']
        mask = (aspect_ratio["min"] <= ar_data) & (ar_data <= aspect_ratio["max"])

        dataset = dataset[mask]

        dataset.dropna(inplace=True)

        for index, data in dataset.iterrows():
            image_name = sub(r"http.+/", "", data["file_url"])
            save_image_to = Path(save_dir) / f"{data["id"]:07d}-{image_name}"

            if save_image_to.exists():
                print(f"[INFO] {save_image_to.name} already exists! Skip. Next URL...")
                continue

            _download(data["file_url"], save_image_to, max_retry, timeout, backoff_factor)
            print(f"{data['id']}-{image_name}")

