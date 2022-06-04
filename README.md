<p align="center">
  <img width="200" height="200" src="https://github.com/magi-1/lattice/blob/main/images/logo.png">
</p>


### Setup

First you must [set up pdm](https://pdm.fming.dev/) after which you can clone this repo and run `pdm install`.

### Notes

- [ ] Use global ftx data stream to inform ftx us
- [ ] Trading rate limiter as part of the investor configuration!
- [ ] Fully functioning limit orders / order cancellation for the local classes.
  - [ ] These orders need to be streamed live (local) market data so they know when to execute. A investor base method that calls a status method at every evaluate_market() time step! Note this is very fast since often you wont have that many orders out at a given time. The status method for a local order will take price as input and trigger the withdrawl from wallet. If it is a live order, it will trigger a withdrawl from the local wallet, since we are emulating the ftx wallet. Good accounting!
- [ ] Wallet config, "balances" paramter no longer optional but creates a ftx subaccount to deploy the strategy on. This would make a subaccount and transfer money from main wallet to the sub account.
- [ ] Add tests for all the feature classes to make sure that the have the correct output dimention (nodes, k)

```yaml
wallet:
    subaccount: 'geezer'
    balance: 10000
```

### Links

- [Things](https://stanford.edu/~ashlearn/)
- [Stanford Qfin](https://stanford.edu/~ashlearn/RLForFinanceBook/chapter9.pdf)
- SeaPearl: [arxiv](https://arxiv.org/pdf/2102.09193v1.pdf) [github](https://github.com/corail-research/SeaPearl.jl)
- [PPO with a good design](https://github.com/google/flax/tree/main/examples/ppo/)
- [Upenn Course](https://gnn.seas.upenn.edu/wp-content/uploads/2020/11/lecture_11_handout.pdf)
- [TGN](https://arxiv.org/pdf/2006.10637.pdf)
- [Trading Algos in Rust](https://github.com/fabianboesiger)
- [Temport GNN Architectures](https://arxiv.org/pdf/2005.11650.pdf)


### Hypermodern

- [click](https://click.palletsprojects.com/en/8.1.x/)
