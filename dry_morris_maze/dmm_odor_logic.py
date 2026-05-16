@returns(tuple)
def process(value):
    flip_state = value.Item1
    trial_phase = value.Item2[3]

    left_air = False
    right_air = False
    fans = 0

    if trial_phase is 'platform':

        if flip_state == 2:
            left_air = False
            right_air = True
            fans = 0
        elif flip_state == 3:
            left_air = True
            right_air = False
            fans = 0

    else:
        left_air = False
        right_air = False
        fans = 128

    return (left_air, right_air, fans)
