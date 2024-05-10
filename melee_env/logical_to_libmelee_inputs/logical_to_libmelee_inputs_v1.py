import melee
from melee_env.logical_inputs.logical_inputs_v1 import logical_inputs_v1 as logical_inputs

def _logical_to_libmelee_single_input_v1(logical_input: logical_inputs, action_repeat_count = 0):
    if logical_input == logical_inputs.joystick_left_up:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 1)
        return libmelee_input
    
    if logical_input == logical_inputs.joystick_up_right:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 1)
        return libmelee_input
    
    if logical_input == logical_inputs.joystick_right_down:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0)
        return libmelee_input
    
    if logical_input == logical_inputs.joystick_down_left:
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

    elif logical_input == logical_inputs.button_A and action_repeat_count == 0:
        def libmelee_input(controller: melee.Controller):
            controller.press_button(melee.enums.Button.BUTTON_A)
        return libmelee_input

    elif logical_input == logical_inputs.button_B and action_repeat_count == 0:
        def libmelee_input(controller: melee.Controller):
            controller.press_button(melee.enums.Button.BUTTON_B)
        return libmelee_input
    
    elif logical_input == logical_inputs.full_shield:
        def libmelee_input(controller: melee.Controller):
            controller.press_shoulder(melee.enums.Button.BUTTON_R, 1)
        return libmelee_input
    
    return None

# requires logical_inputs_v1
def logical_to_libmelee_inputs_v1(logical_inputs_v1, action_repeat_count = 0):    
    libmelee_inputs = []

    for logical_input in logical_inputs_v1:
        libmelee_input = _logical_to_libmelee_single_input_v1(logical_input, action_repeat_count)
        if libmelee_input:
            libmelee_inputs.append(libmelee_input)
    
    return libmelee_inputs

# requires logical_inputs_v1, compatibility function with step_v6
def logical_to_libmelee_inputs_v1_1(logical_inputs_v1, action_repeat_count = 0):    
    libmelee_inputs = []

    class press_nothing():
        def execute(self, controller, character, action_repeat):
            controller.release_all()
    
    libmelee_inputs.append(press_nothing())

    for logical_input in logical_inputs_v1:
        libmelee_input = _logical_to_libmelee_single_input_v1(logical_input, action_repeat_count)
        if libmelee_input:
            class input():
                def execute(self, controller, character, action_repeat):
                    libmelee_input(controller)
                    return True

            libmelee_inputs.append(input())
    
    return libmelee_inputs