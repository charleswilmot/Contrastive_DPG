from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.const import Verbosity
import multiprocessing as mp
import os
from pathlib import Path
from collections import defaultdict
from custom_shapes import TapShape, ButtonShape, LeverShape, Kuka
import numpy as np
from contextlib import contextmanager
from traceback import format_exc
import time
from scipy.interpolate import CubicHermiteSpline
import logging
import sys


MODEL_PATH = os.environ["COPPELIASIM_MODEL_PATH"]


class SimulationConsumerFailed(Exception):
    def __init__(self, consumer_exception, consumer_traceback):
        self.consumer_exception = consumer_exception
        self.consumer_traceback = consumer_traceback

    def __str__(self):
        return '\n\nFROM CONSUMER:\n\n{}'.format(self.consumer_traceback)

def communicate_return_value(method):
    """method from the SimulationConsumer class decorated with this function
    will send there return value to the SimulationProducer class"""
    method._communicate_return_value = True
    return method


def default_dont_communicate_return(cls):
    """Class decorator for the SimulationConsumers meaning that by default, all
    methods don't communicate their return value to the Producer class"""
    for attribute_name, attribute in cls.__dict__.items():
        if callable(attribute):
            communicate = hasattr(attribute, '_communicate_return_value')
            attribute._communicate_return_value = communicate
    return cls


def c2p_convertion_function(cls, method):
    """Function that transform a Consumer method into a Producer method.
    It add a blocking flag that determines whether the call is blocking or not.
    If you call a `Producer.mothod(blocking=False)`, you then must
    `Producer._wait_for_answer()`"""
    def new_method(self, *args, blocking=True, **kwargs):
        cls._send_command(self, method, *args, **kwargs)
        if method._communicate_return_value and blocking:
            return cls._wait_for_answer(self)
    new_method._communicate_return_value = method._communicate_return_value
    return new_method


def consumer_to_producer_method_conversion(cls):
    """Class decorator that transforms all methods from the Consumer to the
    Producer, except for methods starting with an '_', and for the
    multiprocessing.Process methods"""
    proc_methods = [
        "run", "is_alive", "join", "kill", "start", "terminate", "close"
    ]
    method_dict = {
        **SimulationConsumerAbstract.__dict__,
        **SimulationConsumer.__dict__,
    }
    convertables = {
        method_name: method \
        for method_name, method in method_dict.items()\
        if callable(method) and\
        method_name not in proc_methods and\
        not method_name.startswith("_")
    }
    for method_name, method in convertables.items():
        new_method = c2p_convertion_function(cls, method)
        setattr(cls, method_name, new_method)
    return cls


def p2p_convertion_function(name):
    """This function transforms a producer method into a Pool method"""
    def new_method(self, *args, **kwargs):
        if self._distribute_args_mode:
            # all args are iterables that must be distributed to each producer
            for i, producer in enumerate(self._active_producers):
                getattr(producer, name)(
                    *[arg[i] for arg in args],
                    blocking=False,
                    **{key: value[i] for key, value in kwargs.items()}
                )
        else:
            for producer in self._active_producers:
                getattr(producer, name)(*args, blocking=False, **kwargs)
        if getattr(SimulationProducer, name)._communicate_return_value:
            return [
                producer._wait_for_answer() for producer in self._active_producers
            ]
    return new_method

def producer_to_pool_method_convertion(cls):
    """This class decorator transforms all Producer methods (besides close and
    methods starting with '_') to the Pool object."""
    convertables = {
        method_name: method \
        for method_name, method in SimulationProducer.__dict__.items()\
        if callable(method) and not method_name.startswith("_")\
        and not method_name == 'close'
    }
    for method_name, method in convertables.items():
        new_method = p2p_convertion_function(method_name)
        setattr(cls, method_name, new_method)
    return cls


@default_dont_communicate_return
class SimulationConsumerAbstract(mp.Process):
    _id = 0
    """This class sole purpose is to better 'hide' all interprocess related code
    from the user."""
    def __init__(self, process_io, scene="", gui=False):
        super().__init__(
            name="simulation_consumer_{}".format(SimulationConsumerAbstract._id)
        )
        self._id = SimulationConsumerAbstract._id
        SimulationConsumerAbstract._id += 1
        self._scene = scene
        self._gui = gui
        self._process_io = process_io
        np.random.seed()

    def run(self):
        self._pyrep = PyRep()
        self._pyrep.launch(
            self._scene,
            headless=not self._gui,
            verbosity=Verbosity.NONE,
            # verbosity=Verbosity.SCRIPT_ERRORS,
            # verbosity=Verbosity.SCRIPT_INFOS,
            # verbosity=Verbosity.TYRACE_ALL,
            # verbosity=Verbosity.TRACE_LUA,
            # write_coppeliasim_stdout_to_file=True
        )
        self._process_io["simulaton_ready"].set()
        self._main_loop()

    def _close_pipes(self):
        self._process_io["command_pipe_out"].close()
        self._process_io["return_value_pipe_in"].close()
        # self._process_io["exception_pipe_in"].close() # let this one open

    def _main_loop(self):
        success = True
        while success and not self._process_io["must_quit"].is_set():
            success = self._consume_command()
        self._pyrep.shutdown()
        self._close_pipes()

    def _consume_command(self):
        try: # to execute the command and send result
            success = True
            command = self._process_io["command_pipe_out"].recv()
            self._process_io["slot_in_command_queue"].release()
            ret = command[0](self, *command[1], **command[2])
            if command[0]._communicate_return_value:
                self._communicate_return_value(ret)
        except Exception as e: # print traceback, dont raise
            traceback = format_exc()
            success = False # return False: quit the main loop
            self._process_io["exception_pipe_in"].send((e, traceback))
        finally:
            return success

    def _communicate_return_value(self, value):
        self._process_io["return_value_pipe_in"].send(value)

    def signal_command_pipe_empty(self):
        self._process_io["command_pipe_empty"].set()
        while self._process_io["command_pipe_empty"].is_set():
            time.sleep(0.1)

    def good_bye(self):
        pass


@default_dont_communicate_return
class SimulationConsumer(SimulationConsumerAbstract):
    def __init__(self, process_io, scene="", gui=False):
        super().__init__(process_io, scene, gui)
        self._shapes = defaultdict(list)
        self._stateful_shape_list = []
        self._arm_list = []
        self._state_buffer = None
        self._cams = {}

    def set_reset_poses(self):
        self._reset_configuration_trees = [
            arm.get_configuration_tree() for arm in self._arm_list
        ]

    @communicate_return_value
    def reset_pose(self, register_states, register_goals):
        self._previous_hermite_speeds[:] = 0
        self._previous_hermite_accelerations[:] = 0
        for tree in self._reset_configuration_trees:
            self._pyrep.set_configuration_tree(tree)
        self.set_stateful_objects_states(register_states)
        self.set_stateful_objects_goals(register_goals)

    @communicate_return_value
    def reset(self, register_states, register_goals, actions):
        self._previous_hermite_speeds[:] = 0
        self._previous_hermite_accelerations[:] = 0
        for tree in self._reset_configuration_trees:
            self._pyrep.set_configuration_tree(tree)
        self.set_stateful_objects_states(register_states)
        self.set_stateful_objects_goals(register_goals)
        velocities = actions * self._upper_velocity_limits
        self.set_joint_target_velocities(velocities)
        self.step_sim() # three steps with a random velocity for randomization
        self.step_sim()
        self.step_sim()
        return self.get_data()

    @communicate_return_value
    def get_stateful_objects_states(self):
        for i, shape in enumerate(self._stateful_shape_list):
            self._stateful_shape_state_buffer[i] = shape.get_state()
        return self._stateful_shape_state_buffer

    def set_stateful_objects_states(self, states):
        if len(states) != len(self._stateful_shape_list):
            raise ValueError(
            "Can not set the object states, wrong length ({} for {})".format(
                len(states), len(self._stateful_shape_list))
            )
        for shape, state in zip(self._stateful_shape_list, states):
            shape.set_state(state)

    def set_stateful_objects_goals(self, goals):
        if len(goals) != len(self._stateful_shape_list):
            raise ValueError(
            "Can not set the object goals, wrong length ({} for {})".format(
                len(goals), len(self._stateful_shape_list))
            )
        for shape, goal in zip(self._stateful_shape_list, goals):
            shape.set_goal(goal)

    def _add_stateful_object(self, model):
        self._stateful_shape_list.append(model)
        self._stateful_shape_state_buffer = np.zeros(
            len(self._stateful_shape_list),
            dtype=np.uint8
        )

    @communicate_return_value
    def get_state(self):
        n = self._n_joints
        if self._state_buffer is None:
            n_reg = self.get_n_registers()
            size = 3 * n + n_reg
            self._state_buffer = np.zeros(shape=size, dtype=np.float32)
            self._state_mean = np.zeros(shape=size, dtype=np.float32)
            self._state_std = np.zeros(shape=size, dtype=np.float32)
            self._state_mean[3 * n:] = 0.5
            # scaling with values measured from random movements
            pos_std = [1.6, 1.3, 1.6, 1.3, 2.2, 1.7, 2.3]
            spe_std = [1.1, 1.2, 1.4, 1.3, 2.4, 1.7, 2.1]
            for_std = [91, 94, 43, 67, 12, 8.7, 2.3]
            reg_std = [0.5 for i in range(n_reg)]
            self._state_std[0 * n:1 * n] = np.tile(pos_std, n // 7)
            self._state_std[1 * n:2 * n] = np.tile(spe_std, n // 7)
            self._state_std[2 * n:3 * n] = np.tile(for_std, n // 7)
            self._state_std[3 * n:] = reg_std
        self._state_buffer[0 * n:1 * n] = self.get_joint_positions()
        self._state_buffer[1 * n:2 * n] = self.get_joint_velocities()
        try:
            self._state_buffer[2 * n:3 * n] = self.get_joint_forces()
        except RuntimeError:
            self._state_buffer[2 * n:3 * n] = 0
        self._state_buffer[3 * n:] = self.get_stateful_objects_states()
        # STATE NORMALIZATION:
        self._state_buffer -= self._state_mean
        self._state_buffer /= self._state_std
        return self._state_buffer

    @communicate_return_value
    def get_data(self):
        return self.get_state(), self.get_stateful_objects_states()

    @communicate_return_value
    def get_n_registers(self):
        return len(self._stateful_shape_list)

    def add_tap(self, position=None, orientation=None):
        model = self._pyrep.import_model(MODEL_PATH + "/tap_damping_0_spring_20.ttm")
        model = TapShape(model.get_handle(), self._pyrep)
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["tap"].append(model)
        self._add_stateful_object(model)

    def add_button(self, position=None, orientation=None):
        model = self._pyrep.import_model(MODEL_PATH + "/button.ttm")
        model = ButtonShape(model.get_handle(), self._pyrep)
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["button"].append(model)
        self._add_stateful_object(model)

    def add_lever(self, position=None, orientation=None):
        model = self._pyrep.import_model(MODEL_PATH + "/lever_45.ttm")
        model = LeverShape(model.get_handle(), self._pyrep)
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["lever"].append(model)
        self._add_stateful_object(model)

    def add_arm(self, position=None, orientation=None, from_tech_sheet=False):
        if from_tech_sheet:
            model_file = MODEL_PATH + "/kuka_from_tech_sheet.ttm"
        else:
            model_file = MODEL_PATH + "/kuka_default.ttm"
        model = self._pyrep.import_model(model_file)
        model = Kuka(model.get_handle())
        if position is not None:
            model.set_position(position)
        if orientation is not None:
            model.set_orientation(orientation)
        self._shapes["arm"].append(model)
        self._arm_list.append(model)
        self._arm_joints_count = [arm.get_joint_count() for arm in self._arm_list]
        self._n_joints = sum(self._arm_joints_count)
        self._arm_joints_positions_buffer = np.zeros(
            self._n_joints,
            dtype=np.float32
        )
        self._arm_joints_velocities_buffer = np.zeros(
            self._n_joints,
            dtype=np.float32
        )
        self._arm_joints_torques_buffer = np.zeros(
            self._n_joints,
            dtype=np.float32
        )
        self._previous_hermite_speeds = np.zeros(self._n_joints)
        self._previous_hermite_accelerations = np.zeros(self._n_joints)
        self.get_joint_upper_velocity_limits()

    @communicate_return_value
    def add_camera(self, position=None, orientation=None, resolution=[320, 240]):
        vision_sensor = VisionSensor.create(
            resolution=resolution,
            position=position,
            orientation=orientation,
        )
        cam_id = vision_sensor.get_handle()
        self._cams[cam_id] = vision_sensor
        return cam_id

    @communicate_return_value
    def get_frame(self, cam_id):
        return self._cams[cam_id].capture_rgb()

    def delete_camera(self, cam_id):
        self._cams[cam_id].remove()
        self._cams.pop(cam_id)

    @communicate_return_value
    def get_joint_positions(self):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            self._arm_joints_positions_buffer[last:next] = \
                arm.get_joint_positions()
            last = next
        return self._arm_joints_positions_buffer

    @communicate_return_value
    def get_joint_velocities(self):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            self._arm_joints_velocities_buffer[last:next] = \
                arm.get_joint_velocities()
            last = next
        return self._arm_joints_velocities_buffer

    def set_joint_target_velocities(self, velocities):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            arm.set_joint_target_velocities(velocities[last:next])
            last = next

    def set_joint_positions(self, positions):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            arm.set_joint_positions(positions[last:next], disable_dynamics=True)
            last = next

    def set_joint_mode(self, mode):
        for arm in self._arm_list:
            arm.set_joint_mode(mode)

    @communicate_return_value
    def get_joint_mode(self):
        ret = []
        for arm in self._arm_list:
            ret += arm.get_joint_mode(mode)
        return ret

    def set_joint_target_positions(self, positions):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            arm.set_joint_target_positions(positions[last:next])
            last = next

    @communicate_return_value
    def apply_action(self, actions):
        velocities = actions * self._upper_velocity_limits
        self.set_joint_target_velocities(velocities)
        self.step_sim()
        return self.get_data()

    def get_movement_velocities(self, actions, mode='minimalist', span=10):
        if mode == 'minimalist':
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, span))
            velocities = actions * ramp[:, np.newaxis] * self._upper_velocity_limits[np.newaxis]
            velocities = velocities[np.newaxis] # shape [1, span, 7]
        elif mode == "cubic_hermite":
            shape_factor = 0.2
            x = [0, 0.5, 1]
            actions_speeds = actions[:, :2 * self._n_joints]
            actions_speeds = actions_speeds.reshape(
                (2, self._n_joints))
            actions_accelerations = actions[:, 2 * self._n_joints:]
            actions_accelerations = actions_accelerations.reshape(
                (2, self._n_joints))
            speeds = np.vstack([self._previous_hermite_speeds, actions_speeds])
            accelerations = np.vstack([self._previous_hermite_accelerations, actions_accelerations])
            speeds[-1] *= shape_factor
            accelerations[-1] *= shape_factor
            eval = np.linspace(0, 1, span)
            poly = CubicHermiteSpline(x, speeds, accelerations)
            velocities = poly(eval) * self._upper_velocity_limits[np.newaxis]
            velocities = velocities[np.newaxis] # shape [1, span, 7]
            self._previous_hermite_speeds = speeds[-1]
            self._previous_hermite_accelerations = accelerations[-1]
        elif mode == "full_raw":
            velocities = actions * self._upper_velocity_limits[np.newaxis]
            velocities = velocities[:, np.newaxis] # shape [span, 1, 7]
        elif mode == "one_raw":
            velocities = actions * self._upper_velocity_limits[np.newaxis]
            velocities = velocities[np.newaxis] # shape [1, 1, 7]
        else:
            raise ValueError("Unrecognized movement mode ({})".format(mode))
        return velocities

    @communicate_return_value
    def apply_movement(self, actions, mode='minimalist', span=10):
        velocities = self.get_movement_velocities(actions, mode=mode, span=span) # shape [n_states_to_be_returned, mini_sequence_length, n_joints]
        normalized_velocities = velocities / self._upper_velocity_limits[np.newaxis]
        metabolic_costs = np.sum(normalized_velocities ** 2, axis=(1, 2)) # shape [n_states_to_be_returned]
        states_sequence = []
        stateful_objects_states_sequence = []
        for mini_sequence in velocities:
            state, stateful_objects_state = self.get_data()
            states_sequence.append(np.copy(state))
            stateful_objects_states_sequence.append(np.copy(stateful_objects_state))
            for velocity in mini_sequence:
                self.set_joint_target_velocities(velocity)
                self.step_sim()
        return np.vstack(states_sequence), np.vstack(stateful_objects_states_sequence), metabolic_costs

    @communicate_return_value
    def apply_movement_get_frames(self, actions, cam_id, mode='minimalist', span=10):
        velocities = self.get_movement_velocities(actions, mode=mode, span=span)
        normalized_velocities = velocities / self._upper_velocity_limits[np.newaxis]
        metabolic_costs = np.sum(normalized_velocities ** 2, axis=(1, 2)) # shape [n_states_to_be_returned]
        states_sequence = []
        stateful_objects_states_sequence = []
        frames = []
        for mini_sequence in velocities:
            state, stateful_objects_state = self.get_data()
            states_sequence.append(np.copy(state))
            stateful_objects_states_sequence.append(np.copy(stateful_objects_state))
            for velocity in mini_sequence:
                frames.append(self.get_frame(cam_id))
                self.set_joint_target_velocities(velocity)
                self.step_sim()
        return np.vstack(states_sequence), np.vstack(stateful_objects_states_sequence), metabolic_costs, np.array(frames)

    def set_control_loop_enabled(self, bool):
        for arm in self._arm_list:
            arm.set_control_loop_enabled(bool)

    def set_motor_locked_at_zero_velocity(self, bool):
        for arm in self._arm_list:
            arm.set_motor_locked_at_zero_velocity(bool)

    @communicate_return_value
    def get_joint_forces(self):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            self._arm_joints_torques_buffer[last:next] = \
                arm.get_joint_forces()
            last = next
        return self._arm_joints_torques_buffer

    def set_joint_forces(self, forces):
        last = 0
        next = 0
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            arm.set_joint_forces(forces[last:next])
            last = next

    @communicate_return_value
    def get_joint_upper_velocity_limits(self):
        last = 0
        next = 0
        self._upper_velocity_limits = np.zeros(self._n_joints, dtype=np.float32)
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            self._upper_velocity_limits[last:next] = \
                arm.get_joint_upper_velocity_limits()
            last = next
        return self._upper_velocity_limits

    @communicate_return_value
    def get_joint_intervals(self):
        last = 0
        next = 0
        self._intervals = np.zeros((self._n_joints, 2), dtype=np.float32)
        for arm, joint_count in zip(self._arm_list, self._arm_joints_count):
            next += joint_count
            _, self._intervals[last:next] = \
                arm.get_joint_intervals()
            last = next
        return self._intervals

    @communicate_return_value
    def get_n_joints(self):
        return self._n_joints

    def create_environment(self, type='one_arm_4_buttons'):
        if type == 'one_arm_4_buttons':
            self.add_arm()
            distance = 0.65
            self.add_button(position=( distance, 0, 0))
            self.add_button(position=(-distance, 0, 0))
            self.add_button(position=(0,  distance, 0))
            self.add_button(position=(0, -distance, 0))
        elif type == 'one_arm_4_buttons_45':
            self.add_arm()
            distance = 0.65
            sqrt2_2 = 0.7071
            self.add_button(position=( distance * sqrt2_2,  distance * sqrt2_2, 0))
            self.add_button(position=(-distance * sqrt2_2, -distance * sqrt2_2, 0))
            self.add_button(position=(-distance * sqrt2_2,  distance * sqrt2_2, 0))
            self.add_button(position=( distance * sqrt2_2, -distance * sqrt2_2, 0))
        elif type == 'one_arm_4_buttons_near':
            self.add_arm()
            distance = 0.45
            self.add_button(position=( distance, 0, 0))
            self.add_button(position=(-distance, 0, 0))
            self.add_button(position=(0,  distance, 0))
            self.add_button(position=(0, -distance, 0))
        elif type == 'one_arm_8_buttons':
            self.add_arm()
            distance = 0.65
            sqrt2_distance = distance / np.sqrt(2)
            self.add_button(position=( distance, 0, 0))
            self.add_button(position=(-distance, 0, 0))
            self.add_button(position=(0,  distance, 0))
            self.add_button(position=(0, -distance, 0))
            self.add_button(position=(sqrt2_distance, sqrt2_distance, 0))
            self.add_button(position=(-sqrt2_distance, sqrt2_distance, 0))
            self.add_button(position=(sqrt2_distance, -sqrt2_distance, 0))
            self.add_button(position=(-sqrt2_distance, -sqrt2_distance, 0))
        elif type == 'one_arm_4_buttons_4_taps':
            self.add_arm()
            distance = 0.65
            sqrt2_distance = distance / np.sqrt(2)
            self.add_button(position=( distance, 0, 0))
            self.add_button(position=(-distance, 0, 0))
            self.add_button(position=(0,  distance, 0))
            self.add_button(position=(0, -distance, 0))
            self.add_tap(position=(sqrt2_distance, sqrt2_distance, 0))
            self.add_tap(position=(-sqrt2_distance, sqrt2_distance, 0))
            self.add_tap(position=(sqrt2_distance, -sqrt2_distance, 0))
            self.add_tap(position=(-sqrt2_distance, -sqrt2_distance, 0))
        elif type == 'one_arm_2_buttons_2_levers':
            self.add_arm()
            distance = 0.65
            self.add_lever(position=( distance, 0, 0))
            self.add_lever(position=(-distance, 0, 0))
            self.add_button(position=(0,  distance, 0))
            self.add_button(position=(0, -distance, 0))
        elif type == 'one_arm_2_buttons_1_levers_1_tap':
            self.add_arm()
            distance = 0.65
            self.add_lever(position=( distance, 0, 0))
            self.add_tap(position=(-distance, 0, 0))
            self.add_button(position=(0,  distance, 0))
            self.add_button(position=(0, -distance, 0))
        elif type == 'one_arm_4_buttons_2_taps_2_levers':
            self.add_arm()
            distance = 0.65
            sqrt2_distance = distance / np.sqrt(2)
            self.add_lever(position=( distance, 0, 0))
            self.add_lever(position=(-distance, 0, 0))
            self.add_tap(position=(0,  distance, 0))
            self.add_tap(position=(0, -distance, 0))
            self.add_button(position=(sqrt2_distance, sqrt2_distance, 0))
            self.add_button(position=(-sqrt2_distance, sqrt2_distance, 0))
            self.add_button(position=(sqrt2_distance, -sqrt2_distance, 0))
            self.add_button(position=(-sqrt2_distance, -sqrt2_distance, 0))
        else:
            raise ValueError("Unrecognized environment type ({})".format(type))

    def step_sim(self):
        self._pyrep.step()

    def start_sim(self):
        self._pyrep.start()

    def stop_sim(self):
        self._pyrep.stop()

    @communicate_return_value
    def get_simulation_timestep(self):
        return self._pyrep.get_simulation_timestep()

    def set_simulation_timestep(self, dt):
        self._pyrep.set_simulation_timestep(dt)



@consumer_to_producer_method_conversion
class SimulationProducer(object):
    def __init__(self, scene="", gui=False):
        self._process_io = {}
        self._process_io["must_quit"] = mp.Event()
        self._process_io["simulaton_ready"] = mp.Event()
        self._process_io["command_pipe_empty"] = mp.Event()
        self._process_io["slot_in_command_queue"] = mp.Semaphore(100)
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["command_pipe_in"] = pipe_in
        self._process_io["command_pipe_out"] = pipe_out
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["return_value_pipe_in"] = pipe_in
        self._process_io["return_value_pipe_out"] = pipe_out
        pipe_out, pipe_in = mp.Pipe(duplex=False)
        self._process_io["exception_pipe_in"] = pipe_in
        self._process_io["exception_pipe_out"] = pipe_out
        self._consumer = SimulationConsumer(self._process_io, scene, gui=gui)
        self._consumer.start()
        self._logger = logging.getLogger(f"Simulation({self._consumer._id: 2d})")
        self._logger.info("consumer {} started".format(self._consumer._id))
        self._closed = False
        # atexit.register(self.close)

    def _get_process_io(self):
        return self._process_io

    def _check_consumer_alive(self):
        if not self._consumer.is_alive():
            self._consumer.join()
            self._logger.critical("### My friend died ;( raising its exception: ###\n")
            self._consumer.join()
            self._closed = True
            exc, traceback = self._process_io["exception_pipe_out"].recv()
            raise SimulationConsumerFailed(exc, traceback)
        return True

    def _send_command(self, function, *args, **kwargs):
        self._process_io["command_pipe_in"].send((function, args, kwargs))
        semaphore = self._process_io["slot_in_command_queue"]
        while not semaphore.acquire(block=False, timeout=0.1):
            self._check_consumer_alive()

    def _wait_for_answer(self):
        while not self._process_io["return_value_pipe_out"].poll(1):
            self._check_consumer_alive()
        answer = self._process_io["return_value_pipe_out"].recv()
        return answer

    def _wait_consumer_ready(self):
        self._process_io["simulaton_ready"].wait()

    def close(self):
        if not self._closed:
            self._logger.debug("Producer closing")
            if self._consumer.is_alive():
                self._wait_command_pipe_empty(timeout=10)
                self._logger.debug("command pipe empty, setting must_quit flag")
                self._process_io["must_quit"].set()
                self._logger.debug("flushing command pipe")
                self.good_bye()
            self._closed = True
            self._logger.debug("succesfully closed, needs to be joined")
        else:
            self._logger.debug("already closed, doing nothing")

    def join(self, timeout=10):
            self._logger.debug(f"joining ({timeout}) ...")
            self._consumer.join(timeout=timeout)
            if self._consumer.exitcode is None:
                self._logger.warning(f"joining ({timeout}) ... failed")
                self._logger.warning("sending SIGTERM")
                self._consumer.terminate()
                self._consumer.join(timeout=timeout)
                self._logger.warning(f"joining ({timeout}) after SIGTERM ...")
                self._consumer.join(timeout=timeout)
                if self._consumer.exitcode is None:
                    self._logger.warning(f"joining ({timeout}) after SIGTERM ... failed")
            else:
                try:
                    self._logger.debug(f"joining ({timeout}) ... joined!")
                    self._logger.info(f"Coppelia closed")
                except LookupError:
                    pass

    def _wait_command_pipe_empty(self, timeout=None):
        self._send_command(SimulationConsumer.signal_command_pipe_empty)
        if not self._process_io["command_pipe_empty"].wait(timeout=timeout):
            self._logger.info(f"Command pipe was not empty after a timeout of {timeout}sec. Exiting without completing all commands")
        else:
            self._process_io["command_pipe_empty"].clear()

    def __del__(self):
        self.close()


@producer_to_pool_method_convertion
class SimulationPool:
    def __init__(self, size, scene="", guis=[]):
        self.size = size
        self._producers = [
            SimulationProducer(scene, gui=i in guis) for i in range(size)
        ]
        self._active_producers_indices = list(range(size))
        self._distribute_args_mode = False
        self.wait_consumer_ready()

    @contextmanager
    def specific(self, list_or_int):
        _active_producers_indices_before = self._active_producers_indices
        indices = list_or_int if type(list_or_int) is list else [list_or_int]
        self._active_producers_indices = indices
        yield
        self._active_producers_indices = _active_producers_indices_before

    @contextmanager
    def distribute_args(self):
        self._distribute_args_mode = True
        yield len(self._active_producers_indices)
        self._distribute_args_mode = False

    def _get_active_producers(self):
        return [self._producers[i] for i in self._active_producers_indices]
    _active_producers = property(_get_active_producers)

    def close(self):
        for producer in self._producers:
            producer.close()
        for producer in self._producers:
            producer.join()

    def wait_consumer_ready(self):
        for producer in self._producers:
            producer._wait_consumer_ready()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    def test_1():
        scene = ""
        simulation = SimulationProducer(scene, gui=True)
        simulation.add_tap(position=(1, 1, 0), orientation=(0, 0, 1))
        simulation.add_tap(position=(2, 1, 0), orientation=(0, 0, 1))
        simulation.add_button(position=(0, 1, 0), orientation=(0, 0, 0))
        simulation.add_button(position=(0, 0, 0), orientation=(0, 0, 0))
        simulation.add_lever(position=(1, 0, 0), orientation=(0, 0, 0))
        simulation.add_lever(position=(2, 0, 0), orientation=(0, 0, 0))
        simulation.start_sim()

        for j in range(1):
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([1, 0, 0, 0, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 1, 0, 0, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 1, 0, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 0, 1, 0, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 0, 0, 1, 0])
            for i in range(100):
                simulation.step_sim()
            simulation.set_stateful_objects_states([0, 0, 0, 0, 0, 1])

        print(simulation.get_stateful_objects_states())

        for i in range(5000):
            simulation.step_sim()
            print(i, end='\r')
        simulation.stop_sim()
        simulation.close()

    def test_2():
        simulations = SimulationPool(32)
        simulations.add_tap(position=(1, 1, 0), orientation=(0, 0, 1))
        simulations.add_tap(position=(2, 1, 0), orientation=(0, 0, 1))
        simulations.add_button(position=(0, 1, 0), orientation=(0, 0, 0))
        simulations.add_button(position=(0, 0, 0), orientation=(0, 0, 0))
        simulations.add_lever(position=(1, 0, 0), orientation=(0, 0, 0))
        simulations.add_lever(position=(2, 0, 0), orientation=(0, 0, 0))
        simulations.start_sim()
        simulations.set_stateful_objects_states([0, 0, 0, 0, 1, 0])
        print(simulations.get_stateful_objects_states())
        with simulations.specific(0):
            simulations.set_stateful_objects_states([0, 0, 0, 0, 1, 1])
        print(simulations.get_stateful_objects_states())
        simulations.stop_sim()
        return simulations

    def test_3():
        import time
        M = 32
        simulations = SimulationPool(M, guis=[])
        simulations.create_environment('one_arm_2_buttons_1_levers_1_tap')
        simulations.start_sim()
        simulations.step_sim()
        print(simulations.get_joint_positions())
        print(simulations.get_joint_velocities())
        print(simulations.get_joint_forces())
        print(simulations.get_joint_upper_velocity_limits())
        N = 1000
        t0 = time.time()
        for i in range(N):
            simulations.step_sim()
        t1 = time.time()
        print("Pool size: {}, {} iteration in {:.3f} sec ({:.3f} it/sec)".format(
            M,
            N * M,
            t1 - t0,
            M * N / (t1 - t0)
        ))
        simulations.stop_sim()
        simulations.close()

    def test_4():
        import time
        pool_size = 1
        simulations = SimulationPool(
            pool_size,
            scene=MODEL_PATH + '/custom_timestep.ttt',
            guis=[0]
        )
        simulations.create_environment('one_arm_2_buttons_1_levers_1_tap')
        dt = 0.05
        simulations.set_simulation_timestep(dt)
        simulations.set_control_loop_enabled(False)
        simulations.start_sim()
        with simulations.specific(0):
            upper_limits = simulations.get_joint_upper_velocity_limits()[0]
            n_joints = simulations.get_n_joints()[0]

        N = 10000

        periods_in_sec = np.random.randint(
            low=2, high=10, size=n_joints)[np.newaxis]
        periods = periods_in_sec / dt
        x = np.arange(N)[:, np.newaxis]
        velocities = np.sin(x / periods * 2 * np.pi) * upper_limits

        states = []
        t0 = time.time()

        for i in range(N):
            simulations.step_sim()
            simulations.set_joint_target_velocities(velocities[i])
            a = simulations.get_joint_forces()
            states += simulations.get_state()

        t1 = time.time()

        print("{} iteration in {:.3f} sec ({:.3f} it/sec)".format(
            N * pool_size,
            t1 - t0,
            N * pool_size / (t1 - t0)
        ))
        print('mean', np.mean(states, axis=0))
        print('std', np.std(states, axis=0))
        simulations.stop_sim()
        simulations.close()

    def test_5():
        import matplotlib.pyplot as plt
        import time
        simulation = SimulationProducer(gui=True)
        simulation.add_arm()
        n_joints = simulation.get_n_joints()
        simulation.set_control_loop_enabled(False)
        simulation.start_sim()
        simulation.step_sim()
        timestep = simulation.get_simulation_timestep()
        upper_limits = simulation.get_joint_upper_velocity_limits()
        # max_forces = simulation.set_joint_forces([100 for i in range(n_joints)])
        # max_forces = simulation.set_joint_forces([320, 320, 176, 176, 110, 40, 40])
        simulation.step_sim()
        max_forces = simulation.get_joint_forces()
        print("timestep", timestep)
        print(upper_limits)
        print(upper_limits * 360 / 2 / np.pi)
        print(max_forces)
        joint_index = 0
        # reset joint to min pos
        velocities = [0] * n_joints
        velocities[joint_index] = -upper_limits[joint_index]
        simulation.set_joint_target_velocities(velocities)
        for i in range(100):
            simulation.step_sim()
        # go full speed to max position
        velocities[joint_index] = upper_limits[joint_index] * 1
        simulation.set_joint_target_velocities(velocities)
        positions = []
        speeds = []
        forces = []
        for i in range(100):
            simulation.step_sim()
            pos = simulation.get_joint_positions()[joint_index]
            spe = simulation.get_joint_velocities()[joint_index]
            foc = simulation.get_joint_forces()[joint_index]
            positions.append(pos)
            speeds.append(spe)
            forces.append(foc)
        plt.plot(positions, 'b', label='position')
        plt.plot(speeds, 'r', label='speed')
        plt.plot(forces, 'g', label='forces')
        plt.plot(np.full(len(speeds), upper_limits[joint_index]), 'k--', alpha=0.2, label='max speed')
        plt.plot(np.full(len(speeds), velocities[joint_index]), 'r--', alpha=0.6, label='target speed')
        plt.legend()
        plt.show()

    def test_6():
        pool_size = 1
        simulations = SimulationPool(
            pool_size,
            scene=MODEL_PATH + '/custom_timestep.ttt',
            guis=[0]
        )
        simulations.create_environment('one_arm_2_buttons_1_levers_1_tap')
        dt = 0.2
        simulations.set_simulation_timestep(dt)
        simulations.set_control_loop_enabled(False)
        simulations.start_sim()
        with simulations.specific(0):
            n_joints = simulations.get_n_joints()[0]

        N = 1000

        actions = np.random.uniform(low=-1, high=1, size=(N, n_joints * 4))

        states = []
        t0 = time.time()

        for action in actions:
            simulations.apply_movement(action, span=10, mode="cubic_hermite")

        t1 = time.time()

        print("{} iteration in {:.3f} sec ({:.3f} it/sec)".format(
            N * pool_size,
            t1 - t0,
            N * pool_size / (t1 - t0)
        ))
        simulations.stop_sim()
        simulations.close()

    def test_7(pool_size):
        simulations = SimulationPool(
            pool_size,
            scene=MODEL_PATH + '/custom_timestep.ttt',
            guis=[]
        )
        simulations.create_environment('one_arm_2_buttons_1_levers_1_tap')
        dt = 0.2
        simulations.set_simulation_timestep(dt)
        simulations.set_control_loop_enabled(False)
        simulations.start_sim()
        with simulations.specific(0):
            n_joints = simulations.get_n_joints()[0]

        N = 720 // pool_size

        actions = np.random.uniform(low=-1, high=1, size=(N, n_joints))

        t0 = time.time()

        for i, action in enumerate(actions):
            print(i)
            simulations.apply_movement(action, span=10, mode="minimalist")

        t1 = time.time()

        print("{} iteration in {:.3f} sec ({:.3f} it/sec)".format(
            N * pool_size,
            t1 - t0,
            N * pool_size / (t1 - t0)
        ))
        simulations.stop_sim()
        simulations.close()

    def test_8():
        simulation = SimulationProducer(
            scene='',
            gui=False
        )
        simulation.create_environment('one_arm_2_buttons_1_levers_1_tap')
        simulation.start_sim()
        simulation.step_sim()
        state = simulation.get_state()
        print(state.shape)
        simulation.stop_sim()
        simulation.close()

    def test_9(mode='minimalist'):
        np.set_printoptions(precision=3, linewidth=120, suppress=True, sign=' ')
        simulation = SimulationProducer(
            scene=MODEL_PATH + '/custom_timestep.ttt',
            gui=False
        )
        simulation.create_environment('one_arm_2_buttons_1_levers_1_tap')
        dt = 0.05
        span = int(2 / dt)
        print('span', span)
        simulation.set_simulation_timestep(dt)
        simulation.set_control_loop_enabled(False)
        simulation.start_sim()
        simulation.step_sim()
        n_joints = simulation.get_n_joints()
        simulation.apply_movement(np.zeros((1, n_joints)), span=span, mode='minimalist')
        simulation.step_sim()

        N = 10
        if mode == 'minimalist':
            actions = np.random.uniform(low=-1, high=1, size=(N, 1, n_joints))
        elif mode == 'cubic_hermite':
            actions = np.random.uniform(low=-1, high=1, size=(N, 1, n_joints * 4))
        elif mode == 'full_raw':
            actions = np.random.uniform(low=-1, high=1, size=(N, span, n_joints))

        t0 = time.time()

        for i, action in enumerate(actions):
            states, current_goals, metabolic_cost = simulation.apply_movement(action, span=span, mode=mode)
            print(i)
            print(states, current_goals)

        t1 = time.time()

        print("{} iteration in {:.3f} sec ({:.3f} it/sec)".format(N, t1 - t0, N / (t1 - t0)))
        simulation.stop_sim()
        simulation.close()

    def open_one_environment():
        pool_size = 1
        simulation = SimulationProducer(
            scene=MODEL_PATH + '/custom_timestep.ttt',
            gui=True
        )
        simulation.create_environment('one_arm_2_buttons_1_levers_1_tap')
        dt = 0.2
        simulation.set_simulation_timestep(dt)
        simulation.set_control_loop_enabled(False)
        simulation.start_sim()
        simulation.add_camera(
            position=(1.15, 1.35, 1),
            orientation=(
                24 * np.pi / 36,
                -7 * np.pi / 36,
                 4 * np.pi / 36
            ),
            resolution=[320, 240]
        )
        while True:
            simulation.step_sim()
        simulation.stop_sim()
        simulation.close()

    def take_picture(env_name):
        simulation = SimulationProducer(
            scene=MODEL_PATH + '/custom_timestep.ttt',
            gui=False
        )
        simulation.create_environment(env_name)
        simulation.set_control_loop_enabled(False)
        simulation.start_sim()
        cam_id = simulation.add_camera(
            position=(1.15, 1.35, 1),
            orientation=(
                24 * np.pi / 36,
                -7 * np.pi / 36,
                 4 * np.pi / 36
            ),
            resolution=[1920, 1080]
        )
        simulation.step_sim()
        frame = simulation.get_frame(cam_id)
        simulation.stop_sim()
        simulation.close()
        from PIL import Image
        Image.fromarray((frame * 255).astype(np.uint8)).save('/tmp/coppeliasim_frame_{}.png'.format(env_name))

    # open_one_environment()
    # test_9(mode='minimalist')
    # test_9(mode='full_raw')
    take_picture('one_arm_2_buttons_1_levers_1_tap')
    take_picture('one_arm_4_buttons')
