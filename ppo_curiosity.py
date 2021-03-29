import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import pandas as pd

import ray
from ray import tune
from ray.tune import grid_search, analysis
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.tf.attention_net import GTrXLNet
from ray.tune.registry import register_env
from lib import data, environ
import shutil
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.25)

args = parser.parse_args()

def env_creator(env_name):
    if env_name == "StocksEnv-v0":
        from lib.environ import StocksEnv as env
    else:
        raise NotImplementedError
    return env

# register the env
BARS_COUNT = 30
STOCKS = 'crypto_prices__min_train_BITCOIN.csv'
stock_data = {"NIO": data.load_relative(STOCKS)}
env = env_creator("StocksEnv-v0")
tune.register_env('myEnv', lambda config: env(stock_data, bars_count=BARS_COUNT, state_1d=False))

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

class Training(object):
    def __init__(self):
        ray.shutdown()
        ray.init(num_cpus=16, num_gpus=0, ignore_reinit_error=True)
        ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel)
        #ModelCatalog.register_custom_model("attention_net", GTrXLNet)
        self.run = args.run
        self.config_model = {
            "env": "myEnv",  
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "my_model",
                #"conv_filters": None,
                #"conv_activation": "relu",
                #"use_lstm": True,
                #"lstm_use_prev_action": True,
                #"lstm_use_prev_reward": True,
                "vf_share_layers": False,
            },
            "batch_mode": "truncate_episodes",
            "sgd_minibatch_size": 32,
            "num_sgd_iter": 10,
            "lr": 3e-3,  # try different lrs
            "num_workers": 1,  # parallelism
            "framework": "torch",
        }
        self.config_model["num_workers"] = 0
        self.config_model["exploration_config"] = {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": .001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 288,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
            },
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }

        self.stop = {
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
            "episode_reward_mean": args.stop_reward,
        }
    
    def sharpe(self, returns, freq=30, rfr=0):
        """ Given a set of returns, calculates naive (rfr=0) sharpe. """
        eps = np.finfo(np.float32).eps
        return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)
    
    def max_drawdown(self, returns):
        """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
        eps = np.finfo(np.float32).eps
        peak = returns.max()
        trough = returns[returns.argmax():].min()
        return (trough - peak) / (peak + eps)

    def train(self):
        """
        Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # make directory for saves
        # init directory in which to save checkpoints
        saves_root = "saves"
        shutil.rmtree(saves_root, ignore_errors=True, onerror=None)

        # init directory in which to log results
        ray_results = "{}/ray_results/".format(os.getenv("HOME"))
        shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

        analysis = ray.tune.run(
            ray.rllib.agents.ppo.PPOTrainer,
            config=self.config_model,
            local_dir=saves_root,
            stop=self.stop,
            checkpoint_at_end = True)

        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(
            trial=analysis.get_best_trial(
                'episode_reward_mean',
                mode="max",
                scope="all",
                filter_nan_and_inf=True),
                metric='episode_reward_mean')
        # retrieve the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        
        return checkpoint_path, analysis
    

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = ray.rllib.agents.ppo.PPOTrainer(config=self.config_model, env="myEnv")
        self.agent.restore(path)
    
    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env 
        STOCKS = 'stock_prices__min_train_RIOT.csv'
        stock_data = {"RIOT": data.load_relative(STOCKS)}
        env = environ.StocksEnv(
            stock_data,
            bars_count=30,
            reset_on_close=False,
            commission=0.00,
            state_1d=False,
            random_ofs_on_reset=False,
            reward_on_close=True,
            volumes=False)

        episode_reward = 0
        total_steps = 0
        rewards = []

        obs = env.reset()
        while True:
            action = self.agent.compute_action(obs)
            obs, reward, done, _ = env.step(action)
            print("done", done)
            episode_reward += reward
            total_steps += 1
            rewards.append(episode_reward)
            print("{}: reward={} action={}".format(total_steps, episode_reward, action))
            if done:
                break
        
        rewards_data = pd.DataFrame(rewards)
        
        print("Sharpe", rewards_data.apply(self.sharpe, freq=125, rfr=0))

        print("Max Drawdown", rewards_data.apply(self.max_drawdown))

        # plot rewards
        plt.clf()
        plt.plot(rewards)
        plt.title("Total reward, data=RIOT")
        plt.ylabel("Reward, %")
        plt.savefig("ppo_curiousity_model_test_RIOT_30.png")
    

    
if __name__ == "__main__":
    checkpoint_path = "ppo_model_batch30/PPO_2021-03-16_16-05-22/PPO_myEnv_16183_00000_0_2021-03-16_16-05-22/checkpoint_25/checkpoint-25"
    training = Training()
    # Train and save 
    #checkpoint_path, results = training.train()
    # Load saved
    training.load(checkpoint_path)
    # Test loaded
    training.test()