import melee
from melee_env.logical_inputs.logical_inputs_v2 import logical_inputs_v2 as logical_inputs

def _logical_to_libmelee_single_input_v2(logical_input: logical_inputs):
    if logical_input == logical_inputs.joystick_left_up:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 1)
        return libmelee_input
    
    elif logical_input == logical_inputs.joystick_up_right:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 1)
        return libmelee_input
    
    elif logical_input == logical_inputs.joystick_right_down:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0)
        return libmelee_input
    
    elif logical_input == logical_inputs.joystick_down_left:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 0)
        return libmelee_input
    
    elif logical_input == logical_inputs.joystick_left:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 0.5)
        return libmelee_input

    elif logical_input == logical_inputs.joystick_right:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0.5)
        return libmelee_input

    elif logical_input == logical_inputs.joystick_up:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 1)
        return libmelee_input

    elif logical_input == logical_inputs.joystick_down:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0)
        return libmelee_input

    elif logical_input == logical_inputs.button_A:
        def libmelee_input(controller: melee.Controller):
            controller.press_button(melee.enums.Button.BUTTON_A)
        return libmelee_input

    elif logical_input == logical_inputs.button_B:
        def libmelee_input(controller: melee.Controller):
            controller.press_button(melee.enums.Button.BUTTON_B)
        return libmelee_input
    
    elif logical_input == logical_inputs.full_shield:
        def libmelee_input(controller: melee.Controller):
            controller.press_button(melee.enums.Button.BUTTON_R)
        return libmelee_input
    
    return None

# requires logical_inputs_v1
def logical_to_libmelee_inputs_v2(logical_inputs):    
    libmelee_inputs = []

    for logical_input in logical_inputs:
        libmelee_input = _logical_to_libmelee_single_input_v2(logical_input)
        if libmelee_input:
            libmelee_inputs.append(libmelee_input)
    
    return libmelee_inputs