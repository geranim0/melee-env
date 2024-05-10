from melee_env.logical_inputs.logical_inputs_v1 import logical_inputs_v1

# requires data from gamestate_to_obs_v1
def raw_to_logical_inputs_v1(raw_inputs_from_gamestate_to_obs_v1, playerstate = None):
        
    stick_dir = raw_inputs_from_gamestate_to_obs_v1[0]
    buttonA = raw_inputs_from_gamestate_to_obs_v1[1]
    buttonB = raw_inputs_from_gamestate_to_obs_v1[2]
    full_shield = raw_inputs_from_gamestate_to_obs_v1[3]

    logical_actions = []
    if stick_dir == 0:
        logical_actions.append(logical_inputs_v1.joystick_middle)
    
    elif stick_dir == 1:
        logical_actions.append(logical_inputs_v1.joystick_left_up)
    
    elif stick_dir == 3:
        logical_actions.append(logical_inputs_v1.joystick_up_right)

    elif stick_dir == 5:
        logical_actions.append(logical_inputs_v1.joystick_right_down)

    elif stick_dir == 7:
        logical_actions.append(logical_inputs_v1.joystick_down_left)

    elif stick_dir == 8:
        logical_actions.append(logical_inputs_v1.joystick_left)
    
    elif stick_dir == 4:
        logical_actions.append(logical_inputs_v1.joystick_right)

    elif stick_dir == 2:
        logical_actions.append(logical_inputs_v1.joystick_up)
    
    elif stick_dir == 6:
        logical_actions.append(logical_inputs_v1.joystick_down)

    if buttonA == 1:
        logical_actions.append(logical_inputs_v1.button_A)
    
    if buttonB == 1:
        logical_actions.append(logical_inputs_v1.button_B)
    
    if full_shield == 1:
        logical_actions.append(logical_inputs_v1.full_shield)
    
    #print('this frame agent action= ' + str(agent_actions) + ' | ' + str(logical_actions))

    return logical_actions