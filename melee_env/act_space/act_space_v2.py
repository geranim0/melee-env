import gymnasium as gym

def get_action_space_v2(num_players):
    # 2 axis = 3 + 3
    # A, B, shield, modifier_special = 4
    num_joystick_horizontal_positions = 3
    num_joystick_vertical_positions = 3

    if num_players == 2:
        return gym.spaces.MultiDiscrete([num_joystick_horizontal_positions, num_joystick_vertical_positions, 2, 2, 2, 2])
    elif num_players == 4:
        return NotImplementedError("num players must be 2. Got: " + str(num_players))
    else:
        return NotImplementedError("num players must be 2. Got: " + str(num_players))