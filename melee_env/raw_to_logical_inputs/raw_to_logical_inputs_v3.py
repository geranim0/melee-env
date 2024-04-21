from melee_env.logical_inputs.logical_inputs_v1 import logical_inputs_v1

# requires data from gamestate_to_obs_v1
def raw_to_logical_inputs_v3(raw_inputs_from_gamestate_to_obs_v1):
        
    stick_dir = raw_inputs_from_gamestate_to_obs_v1

    logical_actions = []
    if stick_dir == 0:
        logical_actions.append(logical_inputs_v1.joystick_middle)
    
    elif stick_dir == 1:
        logical_actions.append(logical_inputs_v1.joystick_left)
    
    elif stick_dir == 2:
        logical_actions.append(logical_inputs_v1.joystick_up)

    elif stick_dir == 3:
        logical_actions.append(logical_inputs_v1.joystick_right)

    elif stick_dir == 4:
        logical_actions.append(logical_inputs_v1.joystick_down)

    
    #print('this frame agent action= ' + str(agent_actions) + ' | ' + str(logical_actions))

    return logical_actions