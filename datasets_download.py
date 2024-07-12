import sys
sys.path.append("CMC_utils")

import os
import io
import zipfile
import requests
import pandas as pd

import hydra
from omegaconf import DictConfig
import logging

from CMC_utils import save_load


log = logging.getLogger(__name__)


def URL_download(URL: str, path: str, files_to_extract: list) -> None:

    # Download the ZIP file
    response = requests.get(URL)
    response.raise_for_status()  # Check if the download was successful

    # Use BytesIO for the ZIP file downloaded into memory
    zip_file = io.BytesIO(response.content)

    # Base directory where files will be extracted
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Extract specific files from the ZIP file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            # Check if the file is one of the ones we want to extract
            if any(to_extract in file_name for to_extract in files_to_extract):
                # Extract and save the file
                with zip_ref.open(file_name) as zf, open(os.path.join(dir_name, os.path.basename(file_name)), 'wb') as f:
                    f.write(zf.read())


def join_ADULT_sets(path: str, files: list) -> None:
    dfs = []
    dir_name = os.path.dirname(path)
    for file in files:

        df = save_load.load_table(os.path.join(dir_name, file), skiprows=int(file.endswith("test")), header=None)

        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: str(x).strip().replace(".", ""))
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)

    for file in files:
        os.remove(os.path.join(dir_name, file))

    save_load.save_table(dfs, os.path.basename(path), dir_name, header=False)


@hydra.main(version_base="v1.3", config_path="./confs", config_name="datasets_download")
def main(cfg: DictConfig) -> None:
    selected_dbs = cfg.dbs.keys()

    if "adult" in selected_dbs and not os.path.exists(cfg.dbs.adult.path):
        URL_download(URL='https://archive.ics.uci.edu/static/public/2/adult.zip', path=cfg.dbs.adult.path, files_to_extract=['adult.data', 'adult.test'])
        join_ADULT_sets(cfg.dbs.adult.path, ['adult.data', 'adult.test'])
        log.info("Adult dataset downloaded")

    if "bankmarketing" in selected_dbs and not os.path.exists(cfg.dbs.bankmarketing.path):
        URL_download(URL='http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip', path=cfg.dbs.bankmarketing.path, files_to_extract=['bank-additional-full.csv'])
        log.info("Bankmarketing dataset downloaded")

    if "onlineshoppers" in selected_dbs and not os.path.exists(cfg.dbs.onlineshoppers.path):
        URL_download(URL='https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip', path=cfg.dbs.onlineshoppers.path, files_to_extract=['online_shoppers_intention.csv'])
        log.info("OnlineShoppers dataset downloaded")

    if "seismicbumps" in selected_dbs and not os.path.exists(cfg.dbs.seismicbumps.path):
        URL_download(URL="https://archive.ics.uci.edu/static/public/266/seismic+bumps.zip", path=cfg.dbs.seismicbumps.path, files_to_extract=['seismic-bumps.arff'])
        log.info("SeismicBumps dataset downloaded")

    if "spambase" in selected_dbs and not os.path.exists(cfg.dbs.spambase.path):
        URL_download(URL="https://archive.ics.uci.edu/static/public/94/spambase.zip", path=cfg.dbs.spambase.path, files_to_extract=['spambase.data'])
        log.info("Spambase dataset downloaded")


if __name__ == "__main__":
    main()
