""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
from traceback import print_tb
import numpy as np
import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat

class ReorientEnv2V10(BaseV0):
    # Uses a dynamic curriculum, where the environment sets the goal_rot based on the agent's previous performance.
    # Added decay of success threshold to ensure it reaches 0.7 around 1.5 goal_rot, Using uniform distribution for friction etc.

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 100.0,
        "rot_dist": 1.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)

    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            goal_pos = (0.0, 0.0),      # goal position range (relative to initial pos)
            goal_rot = (.785, .785),    # goal rotation range (relative to initial rot)
            obj_size_change = 0,        # object size change (relative to initial size)
            obj_friction_change = (0,0,0),# object friction change (relative to initial size)
            pos_th = .025,              # position error threshold
            rot_th = 0.262,             # rotation error threshold
            drop_th = .200,             # drop height threshold
            trigger_decay = 0.99,       # We add a trigger decay (To exponentially decay the success threshold).
            trigger_num = 20,           # The number of past episodes performance to average 
            trigger_thresh = 0.85,      # The mean success value to achieve for the env to increase difficulty 
            kappa = 1.0,                # Kappa for a von mises distribution. (Not used)
            **kwargs,
        ):
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.goal_init_pos = self.sim.data.site_xpos[self.goal_sid].copy()
        self.goal_obj_offset = self.sim.data.site_xpos[self.goal_sid]-self.sim.data.site_xpos[self.object_sid] # visualization offset between target and object
        self.goal_pos = goal_pos
        self.goal_rot = goal_rot
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th
        self.distance_queue = [1,1]  # To store the sum of error in position + rotation for current and previous time step
        # This is used to implement a reward function as a difference of the distance to the goal. 
        # (There is a minor bug in our implementation here since the initial error can be greater than 1 if the goal_rot is >1.0. However, it did not effect our performance much.)
        self.success_queue = [] # Queue to store past 20 episodes success value
        self.successes = 0 # Variable to accumulate number of states within goal range.
        self.kappa = kappa
        self.trigger_num = trigger_num
        self.trigger_thresh = trigger_thresh
        self.trigger_decay = trigger_decay

        # setup for object randomization
        self.target_gid = self.sim.model.geom_name2id('target_dice')
        self.target_default_size = self.sim.model.geom_size[self.target_gid].copy()

        object_bid = self.sim.model.body_name2id('Object')
        self.object_gid0 = self.sim.model.body_geomadr[object_bid]
        self.object_gidn = self.object_gid0 + self.sim.model.body_geomnum[object_bid]
        self.object_default_size = self.sim.model.geom_size[self.object_gid0:self.object_gidn].copy()
        self.object_default_pos = self.sim.model.geom_pos[self.object_gid0:self.object_gidn].copy()
        self.obj_friction_change = obj_friction_change

        self.obj_size_change = {'high':obj_size_change, 'low':-obj_size_change}
        self.obj_friction_range = {'high':self.sim.model.geom_friction[self.object_gid0:self.object_gidn] + obj_friction_change,
                                    'low':self.sim.model.geom_friction[self.object_gid0:self.object_gidn] - obj_friction_change}
        

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos[:-7] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.5 # Palm up

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid]
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos'] - self.goal_obj_offset # correct for visualization offset between target and object
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pos_dist = np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1))
        rot_dist = np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = pos_dist > self.drop_th
        target_dist = pos_dist + rot_dist
        
        self.distance_queue.append(target_dist[0][0])
        if len(self.distance_queue) > 2:
            self.distance_queue.pop(0)

        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist

            # Optional Keys
            ('pos_dist', -1.*pos_dist),
            ('rot_dist', -1.*rot_dist),
            ('survival', 1),
            ('dist_diff', self.distance_queue[1] - self.distance_queue[0]),
            ('drop', -1 if drop else 0),
            # Must keys
            ('act_reg', -1.*act_mag),
            ('sparse', -rot_dist-10.0*pos_dist),
            ('solved', (pos_dist<self.pos_th) and (rot_dist<self.rot_th) and (not drop) ),
            ('done', drop),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        if (pos_dist<self.pos_th) and (rot_dist<self.rot_th) and (not drop):
            self.successes += 1 # We accumulate successes in this variable
        # Sucess Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        return rwd_dict


    def get_metrics(self, paths, successful_steps=5):
        """
        Evaluate paths and report metrics
        """
        num_success = 0
        num_paths = len(paths)

        # average sucess over entire env horizon
        for path in paths:
            # record success if solved for provided successful_steps
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) > successful_steps:
                num_success += 1
        score = num_success/num_paths

        # average activations over entire trajectory (can be shorter than horizon, if done) realized
        effort = -1.0*np.mean([np.mean(p['env_infos']['rwd_dict']['act_reg']) for p in paths])

        metrics = {
            'score': score,
            'effort':effort,
            }
        return metrics

    def increase_difficulty(self):
        prev_goal = self.goal_rot[1]
        goal_rot = min(self.goal_rot[1] * 1.1, 3.14)       
        self.trigger_thresh = max(self.trigger_thresh * self.trigger_decay, 0.3)

        if prev_goal!=goal_rot:
            self.goal_rot = (-goal_rot, goal_rot)
            print("changing goal ", self.goal_rot, self.trigger_thresh)
            

    def reduce_kappa(self):
        self.kappa = max(self.kappa*0.95,0.5)
        print("changing kappa", self.kappa)

    def set_difficulty(self, goal_rot, trigger_thresh, trigger_num):
        self.goal_rot = (-goal_rot, goal_rot)
        self.trigger_thresh = trigger_thresh
        self.trigger_num = trigger_num
        print("changing goal ", self.goal_rot, self.trigger_thresh, self.trigger_num)
    
    def get_difficulty(self):
        return self.goal_rot[1]

    def set_kappa(self, kappa):
        self.kappa = kappa
        print("changing kappa", self.kappa)


    def reset(self):
        success_ratio = self.successes / self.horizon
        if len(self.success_queue) > self.trigger_num:
            self.success_queue.pop(0) # Maintain queue length to be trigger_num. 
            #There is a minor bug here as well, as immediately after increasing the threshold the queue is not emptied. 
            # But it works fine nevertheless.
        self.success_queue.append(success_ratio)

        if np.mean(self.success_queue) >= self.trigger_thresh:
            self.increase_difficulty()
        self.successes = 0 # Reset success counter.
        self.distance_queue = [1, 1] # Again there is a minor bug in this intial value but it works fine. 

        self.sim.model.body_pos[self.goal_bid] = self.goal_init_pos + \
            self.np_random.uniform( high=self.goal_pos[1], low=self.goal_pos[0], size=3)

        self.sim.model.body_quat[self.goal_bid] = \
            euler2quat(self.np_random.uniform(high=self.goal_rot[1], low=self.goal_rot[0], size=3))

        # Die friction changes
        self.sim.model.geom_friction[self.object_gid0:self.object_gidn] = self.np_random.uniform(**self.obj_friction_range)

        # Die and Target size changes
        del_size = self.np_random.uniform(**self.obj_size_change)
        # adjust size of target
        self.sim.model.geom_size[self.target_gid] = self.target_default_size + del_size
        # adjust size of die
        self.sim.model.geom_size[self.object_gid0:self.object_gidn-3][:,1] = self.object_default_size[:-3][:,1] + del_size
        self.sim.model.geom_size[self.object_gidn-3:self.object_gidn] = self.object_default_size[-3:] + del_size
        # adjust boundary of die
        object_gpos = self.sim.model.geom_pos[self.object_gid0:self.object_gidn]
        self.sim.model.geom_pos[self.object_gid0:self.object_gidn] = object_gpos/abs(object_gpos+1e-16) * (abs(self.object_default_pos) + del_size)

        obs = super().reset()
        return obs