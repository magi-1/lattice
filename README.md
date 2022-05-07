<p align="center">
  <img width="200" height="200" src="https://github.com/magi-1/lattice/blob/main/images/logo.png">
</p>


### Notes

- [ ] Build out base memory buffer functionality into the market class. Have custom Buffer class implimentation
  - [ ] Build out baseline feature / feature set classes which Market classes should expect. These feature classes actually control the memory buffer directly that way we dont have to rewrite market classes for different datasets / asset types / strategies etc. 
- [ ] Make FTXOrder baseclass have api methods, _secret_sign_method() etc. Can just delete the ftx client. This is especially true because I will want to make my own so I can pull higher resolution data. Can have an FTXClient base class that all of these methods can inherit from. Replace the exchanges directory with clients. This will make it easy to write out base api functionality for a number of exchanges without having to have a monolith class. Instead the functionality can incrimentally be built out along with the repo to suit my needs. 
- [ ] resolve the ugliness of `wallet, broker, market` input style. Create a new class to wrap them in that the investor accepts as input.

- [ ] Need to make local market class serve historical orderbook data as well, feature classes to operate on this.
- [ ] Trading rate limiter as part of the investor configuration!
- [ ] Wallet config, "balances" paramter no longer optional but creates a ftx subaccount to deploy the strategy on. This would make a subaccount and transfer money from main wallet to the sub account. The hard part with this is making everything handle sub accounts. 

```yaml
wallet:
    subaccount: 'geezer'
    balance: 10000
```

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
    end
    
    subgraph Market
    id7[Data feed]
    id8[Memory Buffer]
    end

    subgraph Wallet
    id10[Balances]
    id11[I/O]
    end
    
    Wallet --- id1
    Market --- id2
    Broker --- id3
```

```mermaid

flowchart LR

    id1([Random Investor])

    subgraph Exchange Object
    id2([Broker])
    id3([Wallet])
    id4([Market])
    end

    id1-->|Orders| id2
    id3-->|Available Assets| id1
    id4-->|Data| id1
    id2-->|Filled Orders| id3
    id2-.-|Constructor| id1
````

### Links

- SeaPearl: [arxiv](https://arxiv.org/pdf/2102.09193v1.pdf) [github](https://github.com/corail-research/SeaPearl.jl)
- [PPO with a great design](https://github.com/google/flax/tree/main/examples/ppo/)
- [upenn course](https://gnn.seas.upenn.edu/wp-content/uploads/2020/11/lecture_11_handout.pdf)
- [TGN](https://arxiv.org/pdf/2006.10637.pdf)
- [trading algos in rust](https://github.com/fabianboesiger)
