import melee
import gymnasium as gym
import numpy as np

screen_size = 64

# bare minimal rgb
def get_observation_space_v3(num_players):
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


        # int obs
        # jumps_left, percent, stock
        int_obs = gym.spaces.Box(
            low=0,
            high=999,
            shape=(3,),
            dtype=np.float32)

        #float obs
        # position_x, position_y
        float_obs = gym.spaces.Box(
            low=-np.Infinity,
            high=np.Infinity,
            shape=(2,),
            dtype=np.float32)
        
        rgb_obs = gym.spaces.Box(low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8)
        
        if num_players == 2:
            all_obs_space = gym.spaces.Dict({
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,

                'rgb': rgb_obs
            })
        elif num_players == 4:
            all_obs_space = gym.spaces.Dict({
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,
                
                'int_obs_p3': int_obs,
                'float_obs_p3': float_obs,
                
                'int_obs_p4': int_obs,
                'float_obs_p4': float_obs,

                'rgb': rgb_obs
            })
        else:
            raise NotImplementedError('num_players must be 2 or 4')

        return all_obs_space