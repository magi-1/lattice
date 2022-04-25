import yaml
import lattice.paths as paths
import functools


def read_config(config_name):
    with open(paths.configs/f"{config_name}.yaml") as f:
        config = yaml.safe_load(f)
    return config


def check_config(required=[], optional=[]):
    def decorator_func(func):
        @functools.wraps(func)
        def config_check(*args, **kwargs):
            for field in required:
                if field not in kwargs['config']:
                        raise KeyError(f"Failed to specify '{field}' in the config.")
            for field in optional:
                if field not in kwargs['config']:
                    warnings.warn(f"Warning. Did not specify config option {field}.")
            return func(*args, **kwargs)
        return config_check
    return decorator_func


broker_config = check_config()
wallet_config = check_config(required=['balances'])
market_config = check_config(required=['dataset', 'window', 'assets'])