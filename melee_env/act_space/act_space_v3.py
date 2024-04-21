import gymnasium as gym

def get_action_space_v3(num_players):

    num_joystick_positions = 5

    if num_players == 2:
        return gym.spaces.Discrete(num_joystick_positions)
    else:
        return NotImplementedError("num players must be 2 or 4. Got: " + str(num_players))