import gymnasium as gym

# to use with logical_inputs_v3
def get_action_space_v3(num_players):

    num_actions = 28

    if num_players == 2:
        return gym.spaces.Discrete(num_actions)
    else:
        return NotImplementedError("num players must be 2 or 4. Got: " + str(num_players))