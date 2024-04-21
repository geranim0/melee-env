import numpy as np
import melee
import sys

def gamestate_to_obs_v1(gamestate, friendly_ports, enemy_ports, rgb=None):
    # one hot
    action = {port: gamestate.players[port].action for port in gamestate.players.keys()}
    character = {port: gamestate.players[port].character for port in gamestate.players.keys()}
    facing = {port: gamestate.players[port].facing for port in gamestate.players.keys()}
    off_stage = {port: gamestate.players[port].off_stage for port in gamestate.players.keys()}
    on_ground = {port: gamestate.players[port].on_ground for port in gamestate.players.keys()}

    # int obs
    action_frame = {port: gamestate.players[port].action_frame for port in gamestate.players.keys()}
    hitstun_frames_left = {port: gamestate.players[port].hitstun_frames_left for port in gamestate.players.keys()}
    invulnerability_left = {port: gamestate.players[port].invulnerability_left for port in gamestate.players.keys()}
    jumps_left = {port: gamestate.players[port].jumps_left for port in gamestate.players.keys()}
    percent = {port: gamestate.players[port].percent for port in gamestate.players.keys()}
    stock = {port: gamestate.players[port].stock for port in gamestate.players.keys()}

    # float obs
    position = {port: gamestate.players[port].position for port in gamestate.players.keys()}
    shield_strength = {port: gamestate.players[port].shield_strength for port in gamestate.players.keys()}
    speed_air_x_self = {port: gamestate.players[port].speed_air_x_self for port in gamestate.players.keys()}
    speed_ground_x_self = {port: gamestate.players[port].speed_ground_x_self for port in gamestate.players.keys()}
    speed_x_attack = {port: gamestate.players[port].speed_x_attack for port in gamestate.players.keys()}
    speed_y_attack = {port: gamestate.players[port].speed_y_attack for port in gamestate.players.keys()}
    speed_y_self = {port: gamestate.players[port].speed_y_self for port in gamestate.players.keys()}

    current_player_index = 1
    obs = {}
    for port in np.concatenate((friendly_ports, enemy_ports), axis=None):
        
        # one hot
        one_hot_dict_idx = "one_hot_p" + str(current_player_index)

        action_one_hot = np.zeros(len(list(melee.enums.Action)))
        action_one_hot[action[port].value % len(list(melee.enums.Action))] = 1

        if action[port].value >= len(list(melee.enums.Action)):
            print('Error: illegal action: ' + str(action[port].value), file=sys.stderr)

        character_one_hot = np.zeros(len(list(melee.enums.Character)))
        character_one_hot[character[port].value % len(list(melee.enums.Character))] = 1

        if character[port].value >= len(list(melee.enums.Character)):
            print('Error: illegal character: ' + str(action[port].value), file=sys.stderr)

        all_one_hots = np.concatenate((action_one_hot, character_one_hot, facing[port], off_stage[port], on_ground[port]), axis=None).astype(np.float32)
        obs[one_hot_dict_idx] = all_one_hots

        # int
        int_dict_idx = "int_obs_p" + str(current_player_index)
        all_ints = np.concatenate(
            (
                action_frame[port],
                hitstun_frames_left[port],
                invulnerability_left[port],
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
                shield_strength[port],
                speed_air_x_self[port],
                speed_ground_x_self[port],
                speed_x_attack[port],
                speed_y_attack[port],
                speed_y_self[port]
            ), axis=None).astype(np.float32)
        obs[float_dict_idx] = all_floats

        current_player_index += 1
    
    return obs