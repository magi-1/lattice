import lattice.paths as paths
from pathlib import Path
from strictyaml import (
    load, Map, MapCombined, 
    MapPattern, Str, Int, Any,
    Float, Seq, Optional, YAMLError
)


def read_config(config_name):

    schema = Map({
        "wallet": MapCombined({
            "balances": MapPattern(Str(), Float()),
            },
            Str(), 
            Any()
        ),
        "broker": MapCombined({
            Optional("fee"): Float()
            },
            Str(), 
            Any()
        ),
        "market": MapCombined({
            "dataset": Str(),
            "window": Seq(Str()),
            "assets": Seq(Str())
            },
            Str(), 
            Any()
        )
    })

    path = paths.configs/f"{config_name}.yaml"
    with open(path) as f:
        yaml_text = f.read()
        try:
            config = load(yaml_text, schema).data
        except YAMLError as error:
            print(error)
    return config