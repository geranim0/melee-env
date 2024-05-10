
from enum import Enum
from melee_env.logical_inputs.logical_inputs_v3 import logical_inputs_v3 as li
import melee

# requires data from gamestate_to_obs_v1
def raw_to_logical_inputs_v3(raw_input, playerstate):    
    return li(raw_input)