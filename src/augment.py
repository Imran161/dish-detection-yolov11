def get_aug_params():
    return {
        'augment': True,
        'auto_augment': None,
        'mosaic': 1.0,
        'mixup':  0.0,
        'cutmix': 0.0,
        'degrees':     10.0,
        'translate':   0.1,
        'scale':       0.5,
        'shear':       2.0,
        'perspective': 0.0,
        'hsv_h':       0.015,
    }
