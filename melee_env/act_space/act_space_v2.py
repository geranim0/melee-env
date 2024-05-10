import gymnasium as gym

def get_action_space_v2(num_players):

    num_actions = 18

    if num_players == 2:
        return gym.spaces.Discrete(num_actions)
    elif num_players == 4:
        return NotImplementedError("num players must be 2. Got: " + str(num_players))
    else:
        return NotImplementedError("num players must be 2. Got: " + str(num_players))