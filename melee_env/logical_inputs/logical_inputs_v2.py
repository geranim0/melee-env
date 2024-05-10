from enum import Enum

class logical_inputs_v2(Enum):
    nothing = 0
    #standing actions
    joystick_up = 1
    short_hop = 2
    joystick_down = 3
    cstick_left = 4 
    cstick_right = 5
    cstick_up = 6
    cstick_down = 7
    A = 8
    joystick_left = 9 
    joystick_right = 10
    walk_left = 11
    walk_right = 12
    wavedash_left = 13 # rm (if state = jumpsquat, dash = wavedash) (if state = ledgegrab, ledgedash)
    wavedash_right = 14
    B = 15
    B_left = 16
    B_right = 17
    B_down = 18
    B_up = 19
    shield = 20
    grab = 21 # if state = shield: A, if state=running: JC grab
    tilt_left = 22
    tilt_right = 23
    tilt_down = 24
    tilt_up = 25
    # taking damage actions, in air - actions
    joystick_up_left = 26
    joystick_up_right = 27
    joystick_bottom_left = 28
    joystick_bottom_right = 29
    sdi_up = 30
    sdi_down = 31
    sdi_left = 32
    sdi_right = 33
    sdi_joystick_up_left = 34
    sdi_joystick_up_right = 35
    sdi_joystick_bottom_left = 36
    sdi_joystick_bottom_right = 37
    l_cancel = 38 # should be auto
    roll_left = 39
    roll_right = 40
    nair = 41
    unshield = 42
    jab = 43
    short_hop_left = 44
    short_hop_right = 45
    full_hop_left = 46
    full_hop_right = 47

    
