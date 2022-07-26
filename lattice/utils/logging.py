from lattice.utils import plotting
import lattice.paths as paths
import numpy as np
import jax
import jraph


class ExperienceBuffer:
    def __init__(self):
        self.graphs = []
        self.actions = []
        self.reward = None

    def push(self, graph, actions):
        self.graphs.append(graph)
        self.actions.append(actions)

    def reward_to_go(self, rews, num_markets):
        # credits: spinningup.openai.com
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)

        rtgs = (rtgs-rtgs.mean())/rtgs.std()
        self.reward = np.append(np.repeat(rtgs, num_markets), np.zeros(num_markets))

    def state_action_reward(self):
        graph_batch = jraph.batch(self.graphs)
        actions = np.concatenate(self.actions)
        return graph_batch, actions, self.reward


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
