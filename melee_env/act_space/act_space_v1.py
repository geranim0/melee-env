import gymnasium as gym

def get_action_space_v1(num_players):
    # left, right, up, down and diags + nothing = 9 
    # A, B, full R, = 3
    # => 12
    num_joystick_positions = 9

    if num_players == 2:
        return gym.spaces.MultiDiscrete([num_joystick_positions, 2, 2, 2])
    elif num_players == 4:
        return gym.spaces.MultiDiscrete([num_joystick_positions, num_joystick_positions, 2, 2, 2, 2, 2, 2])
    else:
        return NotImplementedError("num players must be 2 or 4. Got: " + str(num_players))