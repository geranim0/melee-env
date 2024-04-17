from melee_env.logical_inputs.logical_inputs_v2 import logical_inputs_v2

# requires data from gamestate_to_obs_v1
def raw_to_logical_inputs_v2(raw_inputs):
        
    stick_horizontal_dir = raw_inputs[0]
    stick_vertical_dir = raw_inputs[1]
    buttonA = raw_inputs[2]
    buttonB = raw_inputs[3]
    shield = raw_inputs[4]
    modifier_special = raw_inputs[5]

    logical_actions = []

    # joystick pos

    # middle
    if stick_horizontal_dir == 0 and stick_vertical_dir == 0:
        logical_actions.append(logical_inputs_v2.joystick_middle)
    
    # left
    elif stick_horizontal_dir == 1 and stick_vertical_dir == 0:
        logical_actions.append(logical_inputs_v2.joystick_left)
    
    # right
    elif stick_horizontal_dir == 2 and stick_vertical_dir == 0:
        logical_actions.append(logical_inputs_v2.joystick_right)

    # up
    elif stick_horizontal_dir == 0 and stick_vertical_dir == 1:
        logical_actions.append(logical_inputs_v2.joystick_up)

    # down
    elif stick_horizontal_dir == 0 and stick_vertical_dir == 2:
        logical_actions.append(logical_inputs_v2.joystick_down)
    
    # up left
    elif stick_horizontal_dir == 1 and stick_vertical_dir == 1:
        logical_actions.append(logical_inputs_v2.joystick_left_up)

    # up right
    elif stick_horizontal_dir == 2 and stick_vertical_dir == 1:
        logical_actions.append(logical_inputs_v2.joystick_up_right)

    # down left
    elif stick_horizontal_dir == 1 and stick_vertical_dir == 2:
        logical_actions.append(logical_inputs_v2.joystick_down_left)

    # down right
    elif stick_horizontal_dir == 2 and stick_vertical_dir == 2:
        logical_actions.append(logical_inputs_v2.joystick_right_down)
    

    if buttonA == 1:
        logical_actions.append(logical_inputs_v2.button_A)

    if buttonB == 1:
        logical_actions.append(logical_inputs_v2.button_B)

    if shield == 1:
        logical_actions.append(logical_inputs_v2.full_shield)

    #print('this frame agent action= ' + str(agent_actions) + ' | ' + str(logical_actions))

    return logical_actions