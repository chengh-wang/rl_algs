"""Use the unchanged MJLab 1.2 G1 and Go1 flat velocity task factories."""


def velocity_cfg(robot, num_envs=1024, seed=42, evaluation=False):
    if robot == 'g1':
        from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
        cfg = unitree_g1_flat_env_cfg()
    elif robot == 'go1':
        from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
        cfg = unitree_go1_flat_env_cfg()
    else:
        raise ValueError(f'Unknown robot {robot!r}')
    cfg.scene.num_envs = num_envs; cfg.seed = seed
    if evaluation:
        cfg.observations['actor'].enable_corruption = False
        cfg.events = {k:v for k,v in cfg.events.items() if k not in
                      ('push_robot', 'foot_friction', 'encoder_bias', 'base_com')}
        cfg.events['reset_base'].params['pose_range'] = {'z': (.02, .02)}
        cfg.curriculum = {}
        command = cfg.commands['twist']
        command.heading_command = False; command.ranges.heading = None
        command.rel_heading_envs = 0.; command.rel_standing_envs = 0.
        command.resampling_time_range = (1e9, 1e9)
    return cfg
