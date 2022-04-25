import yaml
import lattice.paths as paths


def read_config(config_name)
    with open(paths.configs/f"{config_name}.yaml") as f:
        config = yaml.safe_load(f)
    return config