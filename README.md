# lattice

## Design

```mermaid
flowchart LR

    subgraph Exchange
    id1((Wallet))
    id2((Market))
    id3((Broker))
    end

    subgraph Broker
    id4[Open Orders]
    id5[Order Constructor]
    id6[Logger]
    end
    
    subgraph Market
    id7[Data feed]
    id8[Memory Buffer]
    id9[Logger]
    end

    subgraph Wallet
    id10[Balances]
    id11[I/O]
    end
    
    Wallet --> id1
    Market --> id2
    Broker --> id3
```


### Links

- SeaPearl: [arxiv](https://arxiv.org/pdf/2102.09193v1.pdf) [github](https://github.com/corail-research/SeaPearl.jl)
- [PPO with a great design](https://github.com/google/flax/tree/main/examples/ppo/)
- [upenn course](https://gnn.seas.upenn.edu/wp-content/uploads/2020/11/lecture_11_handout.pdf)
- [TGN](https://arxiv.org/pdf/2006.10637.pdf)
- [trading algos in rust](https://github.com/fabianboesiger)
