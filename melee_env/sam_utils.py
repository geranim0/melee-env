from enum import Enum
import melee
from typing import List

class logical_inputs_v1(Enum):
    joystick_left = 0
    joystick_right = 1
    joystick_up = 2
    joystick_down = 3
    button_A = 4
    button_B = 5
    full_shield = 6


class multi_frame_action():
    def not_implemented():
        return

# converts dreamer inputs to logical (jump, wavedash, etc)
def dreamer_to_logical_v1_inputs(dreamer_inputs):
    return
    

def logical_v1_to_libmelee_inputs(logical_input: logical_inputs_v1):
    if logical_input == logical_inputs_v1.joystick_left:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0, 0.5)
        return libmelee_input

    if logical_input == logical_inputs_v1.joystick_right:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0.5)
        return libmelee_input

    if logical_input == logical_inputs_v1.joystick_up:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 1)
        return libmelee_input

    if logical_input == logical_inputs_v1.joystick_down:
        def libmelee_input(controller: melee.Controller):
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0)
        return libmelee_input

    if logical_input == logical_inputs_v1.button_A:
        def libmelee_input(controller: melee.Controller):
            controller.press_button(melee.enums.Button.BUTTON_A)
        return libmelee_input

    if logical_input == logical_inputs_v1.button_B:
        def libmelee_input(controller: melee.Controller):
            controller.press_button(melee.enums.Button.BUTTON_B)
        return libmelee_input
    
    if logical_input == logical_inputs_v1.full_shield:
        def libmelee_input(controller: melee.Controller):
            controller.press_shoulder(melee.enums.Button.BUTTON_R, 1)
        return libmelee_input


#dumb version without diagonals
def logical_v1_to_libmelee_inputs(logical_inputs: List[logical_inputs_v1]):
    
    libmelee_inputs = []

    for logical_input in logical_inputs:
        libmelee_inputs.append(logical_v1_to_libmelee_inputs(logical_input))
    
    return libmelee_inputs