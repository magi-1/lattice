from lattice.utils import plotting
import lattice.paths as paths


class ExperienceBuffer:
    def __init__(self):
        self.log = {}
        self.reward = None

    def push(self, time, state, action):
        self.log[time] = (state, action)

    def assign_reward(self, reward_data):
        self.reward_data = reward_data


def save_results(investor, name=None):
    # arg: run_id=None maybe add later if needed for RL
    """
    Standardize and build upon this
    """

    # Setting out directory
    save_path = paths.data / "sim_out"
    if name != None:
        save_path /= f"sim_{name}"

    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except:
        msg = "Sim data already exists. Try running 'make clear_sims'"
        raise OSError(msg)

    # Writing data
    history = investor.wallet.get_history()
    history.to_parquet(save_path / "wallet_history.parquet", index=False)

    # Saving visualizations
    plotting.visualize_backtest(history, save_path)
