from abc import ABC
from enum import Enum
import melee
from melee_env.logical_inputs.logical_inputs_v2 import logical_inputs_v2 as li
from melee_env.frame_data import jumpsquat

class action():
    def __init__(self, the_action):
        self.the_action = the_action
        self.frame = 0
    
    def execute(self, controller, character, action_repeat):
        finished = self.the_action.execute(controller, character, self.frame, action_repeat)
        self.frame += 1
        return finished

class press_button_forever():
    def __init__(self, button):
        self.button = button

    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            controller.press_button(self.button)
            return True
        
class release_button():
    def __init__(self, button):
        self.button = button

    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            controller.release_button(self.button)
            return True

class press_button_for_action_repeat():
    def __init__(self, button):
        self.button = button

    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            #print(str(self) + ' :: press button at frame ' + str(frame) + ' , ' + str(self.button))
            controller.press_button(self.button)
            return False
        elif frame >= action_repeat:
            #print(str(self) + ' " :: unpress button at frame ' + str(frame) + ' , ' + str(self.button))
            controller.release_button(self.button)
            return True
        else:
            #print(str(self) + ' " :: nothing button at frame ' + str(frame) + ' , ' + str(self.button))
            return False
    

class press_button_1_frame():
    def __init__(self, button):
        self.button = button

    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            controller.press_button(self.button)
            return False
        elif frame >= 1:
            controller.release_button(self.button)
            return True
        else:
            return True

class move_joystick():
    def __init__(self, joystick, x, y):
        self.joystick = joystick
        self.x = x
        self.y = y
    
    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            controller.tilt_analog(self.joystick, self.x, self.y)
        return True

class move_joystick_1_frame():
    def __init__(self, joystick, x, y):
        self.joystick = joystick
        self.x = x
        self.y = y
    
    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            controller.tilt_analog(self.joystick, self.x, self.y)
            return False
        elif frame >= 1:
            controller.tilt_analog(self.joystick, 0.5, 0.5)
            return True
        else:
            return True

class move_joystick_for_action_repeat():
    def __init__(self, joystick, x, y):
        self.joystick = joystick
        self.x = x
        self.y = y
    
    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            controller.tilt_analog(self.joystick, self.x, self.y)
            return False
        elif frame >= action_repeat:
            controller.tilt_analog(self.joystick, 0.5, 0.5)
            return True
        else:
            return False

class direction(Enum):
    left = 0
    right = 1

wavedash_angles = {
    direction.left: (0, 0.35),
    direction.right: (1, 0.35),
}


class wavedash():
    def __init__(self, direction):
        self.x_dir = wavedash_angles[direction][0]
        self.y_dir = wavedash_angles[direction][1]

    def execute(self, controller, character, frame, action_repeat):
        jumpsquat_len = jumpsquat.jump_squat_data[character]

        if frame == 0:
            controller.press_button(melee.enums.Button.BUTTON_Y)
            return False
        elif frame == jumpsquat_len:
            controller.release_button(melee.enums.Button.BUTTON_Y)
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, self.x_dir, self.y_dir)
            controller.press_button(melee.enums.Button.BUTTON_L)
            return False
        elif frame >= jumpsquat_len + 1:
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5)
            controller.release_button(melee.enums.Button.BUTTON_L)
            return True
        else:
            return False
            
class roll():
    def __init__(self, direction):
        self.direction = direction
    
    def execute(self, controller, character, frame, action_repeat):
        if frame == 0:
            controller.press_button(melee.enums.Button.BUTTON_R)
            return False
        elif frame == 1:
            controller.tilt_analog(melee.enums.Button.BUTTON_C, self.direction.value, 0.5)
        elif frame >= action_repeat - 1:
            controller.release_button(melee.enums.Button.BUTTON_R)
            controller.tilt_analog(melee.enums.Button.BUTTON_C, 0.5, 0.5)
            return True
        else:
            return False

def logical_to_libmelee_inputs_v2(logical_input: li):
    
    ret = []
    match logical_input:
        case li.nothing:
            ret = [move_joystick(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5),
                   move_joystick(melee.enums.Button.BUTTON_C, 0.5, 0.5)]
    
        case li.joystick_left:
            ret = [move_joystick(melee.enums.Button.BUTTON_MAIN, 0, 0.5)]

        case li.joystick_right:
            ret = [move_joystick(melee.enums.Button.BUTTON_MAIN, 1, 0.5)]

        case li.joystick_up:
            ret = [move_joystick(melee.enums.Button.BUTTON_MAIN, 0.5, 1)]

        case li.joystick_down:
            ret = [move_joystick(melee.enums.Button.BUTTON_MAIN, 0.5, 0)]

        case li.A:
            ret = [press_button_1_frame(melee.enums.Button.BUTTON_A)]

        case li.jab:
            ret = [move_joystick(melee.enums.Button.BUTTON_MAIN, 0.5, 0.5),
                   press_button_1_frame(melee.enums.Button.BUTTON_A)]

        case li.B:
            ret = [press_button_for_action_repeat(melee.enums.Button.BUTTON_B)]
        
        case li.B_up:
            ret = [press_button_for_action_repeat(melee.enums.Button.BUTTON_B),
                move_joystick(melee.enums.Button.BUTTON_MAIN, 0.5, 1)]
        
        case li.shield:
            ret = [press_button_forever(melee.enums.Button.BUTTON_R)]
        
        case li.unshield:
            ret = [release_button(melee.enums.Button.BUTTON_R)]
        
        case li.short_hop:
            ret = [press_button_1_frame(melee.enums.Button.BUTTON_Y)]
        
        case li.short_hop_left:
            ret = [press_button_1_frame(melee.enums.Button.BUTTON_Y),
                   move_joystick_for_action_repeat(melee.enums.Button.BUTTON_MAIN, 0, 0.5)]

        case li.short_hop_right:
            ret = [press_button_1_frame(melee.enums.Button.BUTTON_Y),
                   move_joystick_for_action_repeat(melee.enums.Button.BUTTON_MAIN, 1, 0.5)]
            
        case li.full_hop_left:
            ret = [press_button_for_action_repeat(melee.enums.Button.BUTTON_Y),
                   move_joystick_for_action_repeat(melee.enums.Button.BUTTON_MAIN, 0, 0.5)]

        case li.full_hop_right:
            ret = [press_button_for_action_repeat(melee.enums.Button.BUTTON_Y),
                   move_joystick_for_action_repeat(melee.enums.Button.BUTTON_MAIN, 1, 0.5)]

        case li.cstick_left:
            ret = [move_joystick_for_action_repeat(melee.enums.Button.BUTTON_C, 0, 0.5)]
        
        case li.cstick_right:
            ret = [move_joystick_for_action_repeat(melee.enums.Button.BUTTON_C, 1, 0.5)]
        
        case li.cstick_up:
            ret = [move_joystick_for_action_repeat(melee.enums.Button.BUTTON_C, 0.5, 1)]

        case li.cstick_down:
            ret = [move_joystick_for_action_repeat(melee.enums.Button.BUTTON_C, 0.5, 0)]
        
        case li.grab:
            ret = [press_button_1_frame(melee.enums.Button.BUTTON_R),
                   press_button_1_frame(melee.enums.Button.BUTTON_A)]
        
        case li.roll_left:
            ret = [roll(direction.left)]
            
        case li.roll_right:
            ret = [roll(direction.right)]
        
        case li.wavedash_left:
            ret = [wavedash(direction.left)]
        
        case li.wavedash_right:
            ret = [wavedash(direction.right)]

    if ret == []:
        print('Error: ' + str(logical_input) + ' not implemented')

    return [action(a) for a in ret]