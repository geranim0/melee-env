from melee import enums
from melee_env.env_v2 import MeleeEnv_v2
from melee_env.agents.basic import *
import argparse

def agent_actions_to_logical_actions_fn(agent_actions):
    return None

def logical_actions_to_controller_actions_fn(logical_actions):
    return None

def gamestate_to_obs_space_fn(gamestate):
    return None

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default=None, type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

players = [Rest(), NOOP(enums.Character.FOX)]

env = MeleeEnv_v2(args.iso, players, agent_actions_to_logical_actions_fn, logical_actions_to_controller_actions_fn, gamestate_to_obs_space_fn, fast_forward=True, shuffle_controllers_after_each_game=False)

episodes = 10000; reward = 0
env.start()

for episode in range(episodes):
    #gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    obs, done = env.reset()
    while not done:
        obs, reward, done, truncated, infos = env.step(None)
        if reward:
            print(reward)

