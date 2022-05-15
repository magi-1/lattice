import lattice.paths as paths
from typing import NewType
from pathlib import Path
from strictyaml import (
    load, Map, MapCombined, 
    MapPattern, Str, Int, Any,
    Float, Seq, Optional, YAMLError
)


WalletConfig = NewType('WalletConfig', dict)
BrokerConfig = NewType('BrokerConfig', dict)
MarketConfig = NewType('MarketConfig', dict)
InvestorConfig = NewType('InvestorConfig', dict)


def read_config(config_name):

    schema = Map({
        "wallet": MapCombined({
            Optional("balances"): MapPattern(Str(), Float()),
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
            "markets": Seq(Str()),
            "features": MapPattern(Str(), Any()),
            Optional("dataset"): Str(),
            Optional("window"): Seq(Str()),
            Optional("resolution"): Int(),
            Optional("lag"): Int(),
            },
            Str(), 
            Any()
        ),
        "investor": Any()
    })

    path = paths.configs/f"{config_name}.yaml"
    with open(path) as f:
        yaml_text = f.read()
        try:
            config = load(yaml_text, schema).data
        except YAMLError as error:
            print(error)
    return config