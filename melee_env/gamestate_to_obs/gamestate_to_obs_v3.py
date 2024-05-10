import numpy as np
import melee
import multiprocessing.shared_memory as shm
import sys

# with rgb (single env only)
def gamestate_to_obs_v3(gamestate, friendly_ports, enemy_ports, rgb):

    # int obs
    jumps_left = {port: gamestate.players[port].jumps_left for port in gamestate.players.keys()}
    percent = {port: gamestate.players[port].percent for port in gamestate.players.keys()}
    stock = {port: gamestate.players[port].stock for port in gamestate.players.keys()}

    # float obs
    position = {port: gamestate.players[port].position for port in gamestate.players.keys()}

    current_player_index = 1
    obs = {}
    for port in np.concatenate((friendly_ports, enemy_ports), axis=None):
    
        # int
        int_dict_idx = "int_obs_p" + str(current_player_index)
        all_ints = np.concatenate(
            (
                jumps_left[port],
                percent[port],
                stock[port]
            ), axis=None).astype(np.float32)
        obs[int_dict_idx] = all_ints

        # floats
        float_dict_idx = "float_obs_p" + str(current_player_index)
        all_floats = np.concatenate(
            (
                position[port].x,
                position[port].y,
            ), axis=None).astype(np.float32)
        obs[float_dict_idx] = all_floats

        current_player_index += 1
    
    obs['rgb'] = rgb
    
    return obs