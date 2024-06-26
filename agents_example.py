from melee import enums
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
import argparse

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default=None, type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

parser.add_argument("--slippi_game_path", default=None, type=str,
    help="path to slippi appimage")

args = parser.parse_args()

players = [Shine(), NOOP(enums.Character.FOX)]

env = MeleeEnv(args.iso, args.slippi_game_path, players, fast_forward=True)

episodes = 10000; reward = 0
env.start()

for episode in range(episodes):
    gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    while not done:
        for i in range(len(players)):
            players[i].act(gamestate)
        gamestate, reward, done, truncated, infos = env.step()
        if reward and reward[0]:
            print(reward[0])