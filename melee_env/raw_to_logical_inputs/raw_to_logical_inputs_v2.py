
from enum import Enum
from melee_env.logical_inputs.logical_inputs_v2 import logical_inputs_v2 as li
import melee

class states(Enum):
    on_ground = 0
    in_air = 1
    in_shield = 2
    on_ledge = 3

action_maps = {
    states.on_ground:{
        0: li.nothing,
        1: li.joystick_left,
        2: li.joystick_right,
        3: li.joystick_down,
        4: li.joystick_up,
        5: li.short_hop,
        6: li.cstick_left,
        7: li.cstick_right,
        8: li.cstick_down,
        9: li.cstick_up,
        10: li.jab,
        11: li.B,
        12: li.B_up,
        13: li.shield,
        15: li.grab,
        16: li.roll_left,
        17: li.roll_right,
        18: li.wavedash_left,
        19: li.wavedash_right
    },
    states.in_air:{
        0: li.nothing,
        1: li.joystick_left,
        2: li.joystick_right,
        3: li.joystick_down,
        4: li.joystick_up,
        5: li.short_hop,
        6: li.cstick_left,
        7: li.cstick_right,
        8: li.cstick_down,
        9: li.cstick_up,
        10: li.nair,
        11: li.B,
        12: li.B_up,
        13: li.shield,
        14: li.joystick_bottom_left,
        15: li.joystick_bottom_right,
        16: li.joystick_up_left,
        17: li.joystick_up_right,
        18: li.B_up
    },
    states.in_shield: {
        0: li.nothing,
        4: li.joystick_up,
        5: li.short_hop,
        6: li.short_hop_left,
        7: li.short_hop_right,
        8: li.full_hop_left,
        9: li.full_hop_right,
        15: li.grab,
        16: li.roll_left,
        17: li.roll_right,
        18: li.wavedash_left,
        19: li.wavedash_right
        # add short up left, right, big hop left right
    }
    #'on_ledge':{
    #    0: li.nothing,
    #    1: li.joystick_left,
    #    2: li.joystick_right,
    #    3: li.joystick_down,
    #    4: li.joystick_up,
    #    5: li.short_hop,
    #    6: li.A,
    #    7: li.shield,
    #}
    # 'in-shield'
}

# requires data from gamestate_to_obs_v1
def raw_to_logical_inputs_v2(raw_input, playerstate):
        
    on_ground = playerstate.on_ground
    in_shield = playerstate.action == melee.enums.Action.SHIELD
    on_ledge = playerstate.action in [melee.enums.Action.EDGE_HANGING]

    action_map_key = states.on_ground
    if not on_ground:
        action_map_key = states.in_air
    
    if in_shield:
        action_map_key = states.in_shield
    
    #if raw_input < len(action_maps[action_map_key]):
    if raw_input in action_maps[action_map_key]:
        return action_maps[action_map_key][raw_input]
    else:
        return action_maps[action_map_key][0] # nothing