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
    help="Full (not relative) path to your NTSC 1.02/PAL SSBM Melee ISO")

parser.add_argument("--slippi_game_path", default=None, type=str,
    help="path to slippi appimage")

parser.add_argument("--env_num", default=None, type=str, 
    help="if using more than 1 env")

parser.add_argument("--slippi_port", default="51441", type=str, 
    help="if using more than 1 env")

args = parser.parse_args()

#players = [Rest(), NOOP(enums.Character.FOX)]
#players = [Rest(), AgentChooseCharacter(enums.Character.MARTH)]
#players = [Rest(), sam_ai()] # works doenst get stuck in menu
players = [sam_ai(), Rest()] # gets stuck in menu
#players = [Rest(), NOOP(enums.Character.FOX)]
#players = [NOOP(enums.Character.FOX), NOOP(enums.Character.FOX)]
#players = [sam_ai(), CPU(melee.enums.Character.JIGGLYPUFF, 1)]

env = MeleeEnv_v2(args.iso, args.slippi_game_path, players, fast_forward=True, shuffle_controllers_after_each_game=True, num_players=2, action_repeat=12, env_num=args.env_num, slippi_port=args.slippi_port)

episodes = 10000; reward = 0
env.start()

for episode in range(episodes):
    #gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    print('done! resetting')
    obs, done = env.reset()
    while not done:
        simulated_action = env.action_space.sample() 
        obs, reward, done, truncated, infos = env.step(simulated_action)
        if reward:
            print(reward)

