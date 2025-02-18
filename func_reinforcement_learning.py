from stable_baselines3 import PPO
import gym
import numpy as np
import pandas as pd

import gym
import numpy as np
import pandas as pd

class PairSelectionEnv(gym.Env):
    def __init__(self, pairs_data):
        super(PairSelectionEnv, self).__init__()
        
        # Filter out non-numeric columns
        self.pairs_data = pairs_data.select_dtypes(include=[np.number])
        
        # Dynamically define the observation space based on the number of columns in pairs_data
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(pairs_data.columns),)
        )
        self.action_space = gym.spaces.Discrete(len(pairs_data))

    def step(self, action):
        reward = self.pairs_data.iloc[action]['final_score']
        done = True
        return self.pairs_data.iloc[action].values, reward, done, {}

    def reset(self):
        return self.pairs_data.iloc[0].values

    
def train_rl_agent(df_pairs):
    """Train an RL agent to rank pairs."""
    env = PairSelectionEnv(df_pairs)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

def rank_pairs_with_rl(refined_pairs, rl_model):
    """Rank pairs using the trained RL agent."""
    ranked_pairs = []
    for pair in refined_pairs:
        # Use only the available features for the observation
        obs = np.array([
            pair['composite_score'],
            pair['liquidity_score'],
            pair['hurst'],
            1 / pair['half_life']  # Use inverse of half-life for better scaling
            pair('dynols_pvalue')  # Default value if not present
        ])
        action, _ = rl_model.predict(obs)
        ranked_pairs.append((pair, action))
    
    # Sort pairs by action scores in descending order
    ranked_pairs.sort(key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in ranked_pairs]