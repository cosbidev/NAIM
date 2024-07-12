import sys
sys.path.append("CMC_utils")

import hydra
import logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from CMC_utils.pipelines import supervised_learning_main, supervised_tabular_missing_main

log = logging.getLogger(__name__)


@hydra.main(version_base="v1.3", config_path="confs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.paths = instantiate(cfg.paths)

    pipelines_options = dict(simple=supervised_learning_main, missing=supervised_tabular_missing_main)

    pipelines_options[cfg.pipeline](cfg)


if __name__ == '__main__':
    main()
