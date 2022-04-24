# lattice

### Graph Based Reinforcement Learning

- SeaPearl: [arxiv](https://arxiv.org/pdf/2102.09193v1.pdf) [github](https://github.com/corail-research/SeaPearl.jl)

### flax

- [PPO with a great design](https://github.com/google/flax/tree/main/examples/ppo/)

### notes

- [upenn course](https://gnn.seas.upenn.edu/wp-content/uploads/2020/11/lecture_11_handout.pdf)
- [TGN](https://arxiv.org/pdf/2006.10637.pdf)

### interesting

- [trading algos in rust](https://github.com/fabianboesiger)

### python style

```python
from typing import List, NewType,
ListOfDicts = NewType("ListOfDicts", List[dict])
```

# Todos

```python
Investor(LocalWallet, FTXMarket) # local trades
Investor(FTXWallet, LocalMarket) # local trades
Investor(LocalWallet, LocalMarket) # local trades
Investor(FTXWallet, FTXMarket) # remote trades
```

```python
class Investor:
    wallet: Wallet
    market: Market

    def place_order(self, order: Order):
        # The problem lies here
        self.market.place_order() # Fails here Investor(LocalWallet, FTXMarket)
        self.wallet.place_order() # Doesnt make sense and fails here Investor(FTXWallet, LocalMarket)

```


# Research

- Have several tunable technical indicators to optimize over (later on).