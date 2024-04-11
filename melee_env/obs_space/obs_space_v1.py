import melee
import gymnasium as gym
import numpy as np

def get_observation_space_v1(num_players):
        # action (enum.Action): one hot (385 bits)
        # action_frame (int):
        # character (enum.Character): 
        # facing (bool): 
        # hitstun_frames_left (int): 
        # invulnerability_left (int): 
        # jumps_left (int): 
        # off_stage (bool):
        # on_ground (bool): 
        # percent (int):
        # position (float, float): 
        # shield_strength (float max 60): 
        # speed_air_x_self (float): 
        # speed_ground_x_self(float): 
        # speed_x_attack (float): 
        # speed_y_attack (float):
        # speed_y_self (float): 
        # stock (int): 
        # ----- for later ---------
        # stage, projectiles,  

        # one hot encoded obs
        action_len = len(list(melee.enums.Action)) # 385
        character_len = len(list(melee.enums.Character))
        facing_len = 1
        offstage_len = 1
        on_ground_len = 1

        # todo: try multibinary and integer type and see diff. is this wasteful?
        # action, character, facing, off_stage, on_ground
        one_hot_obs = gym.spaces.Box(low=0, high=1, shape=(action_len + character_len + facing_len + offstage_len + on_ground_len,), dtype=np.float32)

        # int obs
        # action_frame, hitstun_frames_left, invulnerability_left, jumps_left, percent, stock
        int_obs = gym.spaces.Box(
            low=0,
            high=999,
            shape=(6,),
            dtype=np.float32)

        #float obs
        # position_x, position_y, shield strength, speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
        float_obs = gym.spaces.Box(
            low=-np.Infinity,
            high=np.Infinity,
            shape=(8,),
            dtype=np.float32)
        

        if num_players == 2:
            all_obs_space = gym.spaces.Dict({
                'one_hot_p1': one_hot_obs,
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'one_hot_p2': one_hot_obs,
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,
            })
        elif num_players == 4:
            all_obs_space = gym.spaces.Dict({
                'one_hot_p1': one_hot_obs,
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'one_hot_p2': one_hot_obs,
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,
                
                'one_hot_p3': one_hot_obs,
                'int_obs_p3': int_obs,
                'float_obs_p3': float_obs,
                
                'one_hot_p4': one_hot_obs,
                'int_obs_p4': int_obs,
                'float_obs_p4': float_obs,
            })
        else:
            raise NotImplementedError('num_players must be 2 or 4')

        return all_obs_space