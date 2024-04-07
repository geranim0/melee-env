from melee import enums
from melee_env.env_v2 import MeleeEnv_v2
from melee_env.agents.basic import *
import argparse
from melee_env import sam_utils
from melee_env.gamestate_to_obs import gamestate_to_obs_v1
from melee_env.raw_to_logical_inputs import raw_to_logical_inputs_v1
from melee_env.logical_to_libmelee_inputs import logical_to_libmelee_inputs_v1

def random_act(obs):
    return env.action_space.sample()

def id(x):
    return x

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
#players = [sam_ai(), Rest()] # gets stuck in menu
#players = [Rest(), NOOP(enums.Character.FOX)]
#players = [NOOP(enums.Character.FOX), NOOP(enums.Character.FOX)]
#players = [sam_ai(), CPU(melee.enums.Character.JIGGLYPUFF, 1)]
players = [step_controlled_ai(gamestate_to_obs_v1.gamestate_to_obs_v1,
              raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
              logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1,
              agent_type.step_controlled_AI),
            trained_ai(id, 
               random_act, 
               raw_to_logical_inputs_v1.raw_to_logical_inputs_v1, 
               logical_to_libmelee_inputs_v1.logical_to_libmelee_inputs_v1,
               agent_type.enemy_controlled_AI)] 

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
