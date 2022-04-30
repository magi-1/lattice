from lattice.investor import BernoulliInvestor, Investor
from lattice.broker import LocalBroker
from lattice.market import LocalMarket
from lattice.wallet import LocalWallet
from lattice.config import read_config
import lattice.paths as paths
import argparse


"""
TURN THIS INTO AN EXAMPLE LIVE BOT, WITH LOCAL ORDERS BUT LIVE MARKET!!!!
Just have a comment saying # Set as FTXOrders to place real trades!!!!
"""


def log_backtest(investor: Investor):
    import lattice.utils.plotting as plot
    history = investor.wallet.get_history()
    history.to_csv(paths.data/'sim_out'/'wallet_history.csv',index=False)
    plot.visualize_backtest(history, paths.data/'sim_out')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Name of experiment"
        )
    args = vars(parser.parse_args())

    config = read_config(args['config'])
    market = LocalMarket(config=config['market'])
    wallet = LocalWallet(config=config['wallet'])
    broker = LocalBroker(config=config['broker'])
    investor = BernoulliInvestor(wallet, market, broker, p=[.5,.5])

    while total_value := investor.evaluate_market():
        print(f'Total Value: ${total_value:.2f}')

    log_backtest(investor)