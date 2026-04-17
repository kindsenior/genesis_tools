import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, quat_to_R
import math
import os
from pathlib import Path
import torch

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Sbr1Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False,  rendered_envs_idx=list(range(1))):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(1.0, 4.0, 0.3),
                camera_lookat=(1.0, 0.0, 0.3),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=rendered_envs_idx),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # enable_self_collision=True,
            ),
            show_viewer=show_viewer,
        )

        # add ground
        self.ground = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        this_file = Path(__file__).resolve() # genesis_tools/scripts/sbr1_locomotion/sbr1_env.py
        pkg_root = this_file.parents[2] # genesis_tools/
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=pkg_root / "models" / "sample_bipedal_robot" / "sbr1.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([kp * rate for kp, rate in zip([self.env_cfg["kp"]] * self.num_actions, self.env_cfg["pdgain_rate"])], self.motor_dofs)
        self.robot.set_dofs_kv([kd * rate for kd, rate in zip([self.env_cfg["kd"]] * self.num_actions, self.env_cfg["pdgain_rate"])], self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.feet_pos = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=gs.tc_float)
        self.feet_quat = torch.zeros((self.num_envs, 2, 4), device=self.device, dtype=gs.tc_float)
        self.feet_lin_vel = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=gs.tc_float)
        self.feet_ang_vel = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=gs.tc_float)
        self.feet_contact_force = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=gs.tc_float)
        self.feet_contacts = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_last_contacts = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        self.obs_scales_pos = torch.tensor(self.obs_scales["pos"], device=self.device, dtype=gs.tc_float)
        self.feet_init_pos_y = torch.tensor([0.125, -0.125], device=self.device, dtype=gs.tc_float)
        self.base_lin_vel_x_th = torch.ones((self.num_envs, 1), device=self.device, dtype=gs.tc_float) * 1.5

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def _check_nan(self, name, tensor):
        if torch.isnan(tensor).any():
            print(f"[NaN DETECTED] {name}: min={tensor.min().item()}, max={tensor.max().item()}")
            print(tensor)

            nan_mask = torch.isnan(tensor).any(dim=1)
            nan_indices = nan_mask.nonzero(as_tuple=False).squeeze()
            print(f"[NaN DETECTED] in {name}")
            print(f"Indices with NaN: {nan_indices.tolist()}")
            print(f"Rows with NaN:")
            print(tensor[nan_mask])
            raise ValueError(f"NaN detected in {name}: {tensor}")

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.feet_pos[:] = self.robot.get_links_pos()[:, [11, 12], :]
        self.feet_quat[:] = self.robot.get_links_quat()[:, [11, 12], :]
        self.feet_lin_vel[:] = self.robot.get_links_vel()[:, [11, 12], :]
        self.feet_ang_vel[:] = self.robot.get_links_ang()[:, [11, 12], :]
        self.feet_contact_force[:] = self.robot.get_links_net_contact_force()[:, [11, 12], :]
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                self.base_pos * self.obs_scales_pos, # 3
            ],
            axis=-1,
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew


        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # add
        self.extras["observations"]["critic"] = self.obs_buf

        # return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras # old
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras # new

    # # old
    # def get_observations(self):
    #     return self.obs_buf
    # new
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # randomization
        self.randomize_friction()
        self.randomize_pd_gains(67, 150, 20, 45)
        self.randomize_armature()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_tracking_lin_vel_x(self):
        slow = self.base_lin_vel[:, 0] < 2.5
        return 2.5 + (self.base_lin_vel[:, 0] - 2.5) * slow

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_base_y(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 1])
    
    def _reward_base_orientation(self):
        R = quat_to_R(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        return (R[:, 2, 2])
    
    def _reward_alive(self):
        return 1.0
    
    def _reward_feet_slip(self):
        contact = torch.norm(self.feet_contact_force, dim=2) > 1
        contact_feet_vel = torch.square(self.feet_lin_vel) * contact.unsqueeze(-1)  # [num_envs, 2, 3]
        contact_feet_ang = torch.square(self.feet_ang_vel) * contact.unsqueeze(-1)  # [num_envs, 2, 3]
        penalize = torch.square(contact_feet_vel) + torch.square(contact_feet_ang)  # [num_envs, 2, 3]
        return torch.sum(penalize, dim=(1, 2))

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = torch.norm(self.feet_contact_force, dim=2) > 1
        contact_filt = torch.logical_or(contact, self.feet_last_contacts) 
        self.feet_last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.feet_contact_force, dim=2) > 1
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.3) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_feet_orientation(self):
        R = quat_to_R(
            transform_quat_by_quat(torch.ones_like(self.feet_quat) * self.inv_base_init_quat, self.feet_quat)
        )
        return torch.sum(R[:, :, 2, 2], dim=1)
    
    def _reward_feet_y(self):
        # return torch.sum(torch.square(self.feet_pos[:, :, 1] - self.feet_init_pos_y), dim=1)
        return torch.sum(torch.abs(self.feet_pos[:, :, 1] - self.feet_init_pos_y) > 0.05, dim=1)

    # ------------ randomization ----------------
    def randomize_link_properties(self):
        # scale mass of links
        mass_scale = 0.9 + 0.2 * torch.rand((self.robot.n_links,), device=self.device)  # 0.9〜1.1
        self.robot.set_links_inertial_mass(mass_scale)

    def randomize_com_shift(self):
        # shift COM positions
        num_links = self.robot.n_links
        link_indices = list(range(num_links))
        com_shift = 0.01 * torch.randn((self.num_envs, num_links, 3), device=self.device)  # +-0.01m=+-10mm
        self.robot.set_COM_shift(com_shift, link_indices)

    def randomize_friction(self):
        # frictions between the ground and robots
        # friction = 0.5 + torch.rand(1).item() # 0.5~1.5
        friction = 0.2 + 1.6*torch.rand(1).item() # 0.2~1.8

        self.robot.set_friction(friction)
        self.ground.set_friction(friction)

    def randomize_pd_gains(self, kp_min=18.0, kp_max=30.0, kv_min=0.7, kv_max=1.5):
        # # pd gains of the joint control
        # num_dofs = self.robot.n_dofs
        # kp_min, kp_max = 18.0, 30.0
        # kv_min, kv_max = 0.7, 1.2
        kp = (torch.rand(self.num_actions, device=self.device) * (kp_max - kp_min) + kp_min) * torch.tensor(self.env_cfg["pdgain_rate"], device=self.device)
        kv = (torch.rand(self.num_actions, device=self.device) * (kv_max - kv_min) + kv_min) * torch.tensor(self.env_cfg["pdgain_rate"], device=self.device)
        self.robot.set_dofs_kp(kp, self.motor_dofs)
        self.robot.set_dofs_kv(kv, self.motor_dofs)

    def randomize_armature(self):
        # joint's rotor inertia
        armature_min, armature_max = 0.01, 0.15
        armature = torch.rand(self.robot.n_dofs, device=self.device) * (armature_max - armature_min) + armature_min
        self.robot.set_dofs_armature(armature)
