import os
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from predicate_learning.runners import create_runner

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def set_random_seed(seed):
    import random
    import numpy as np
    import torch
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    assert len(cfg) > 0, "No config file specified!"

    # set random seed
    set_random_seed(cfg.seed)
    logger.info(f"Random seed set to {cfg.seed}")

    # dump config
    OmegaConf.save(cfg, Path(cfg.save_dir) / "config.yaml")

    runner = create_runner(cfg.runner, cfg)
    runner.run()


if __name__ == "__main__":
    main()
