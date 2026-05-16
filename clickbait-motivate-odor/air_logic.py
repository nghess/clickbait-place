@returns(tuple)
def process(value):
    flip_state = value.Item1
    reward_state = value.Item2[2]
    active_target = value.Item2[3]

    left_air = False
    right_air = False
    fans = 0

    if active_target == 'None':
        active_target = None
    else:
        active_target = int(active_target)

    if active_target is not None:
        if flip_state == 0:
            left_air = True
            right_air = False
            fans = 0
        elif flip_state == 1:
            left_air = False
            right_air = True
            fans = 0
        elif flip_state == 2:
            left_air = False
            right_air = False
            fans = 128

    elif active_target is None and reward_state == True:
        left_air = False
        right_air = False
        fans = 255

    else:
        left_air = False
        right_air = False
        fans = 0

    return (left_air, right_air, fans)
