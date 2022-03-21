import pickle
import logging
from agent import Agent
import numpy as np
from tkinter import *
from  tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from simulation import SimulationProducer
from threading import Thread
from scipy.optimize import minimize, Bounds


def get_action_plane(action, fixed_joints=(0, 1), N=100):
    fixed_joints = tuple(sorted(fixed_joints))
    action = np.array(action)
    j0, j1 = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]
    actions = np.broadcast_to(action[np.newaxis, np.newaxis], (N, N, 7))
    return np.concatenate([
        actions[:, :, :fixed_joints[0]],
        j0[:, :, np.newaxis],
        actions[:, :, (fixed_joints[0] + 1):fixed_joints[1]],
        j1[:, :, np.newaxis],
        actions[:, :, (fixed_joints[1] + 1):],
    ], axis=-1)


class GUIController:
    def __init__(self,
            restore_path,
            discount_factor,
            noise_magnitude_limit,
            hierarchization_config,
            hierarchization_coef,
            actor_learning_rate,
            critic_learning_rate,
            action_dim,
            height=320,
            width=240,
            ):
        self.restore(restore_path)
        self.agent = Agent(
            discount_factor,
            noise_magnitude_limit,
            hierarchization_config,
            hierarchization_coef,
            actor_learning_rate,
            critic_learning_rate,
            action_dim,
        )
        self.simulation = SimulationProducer(
            scene='../3d_models/custom_timestep.ttt')
        self.simulation.set_simulation_timestep(0.2)
        self.simulation.create_environment()
        self.simulation.set_control_loop_enabled(False)
        self.simulation.start_sim()
        self.simulation.set_reset_poses()
        self._cam_id = self.simulation.add_camera(
            position=(1.15, 1.35, 1),
            orientation=(
                24 * np.pi / 36,
                -7 * np.pi / 36,
                 4 * np.pi / 36
            ),
            resolution=(height, width),
        )
        self.simulation.step_sim()
        self.actions_dim = self.simulation.get_n_joints()
        self.registers_dim = self.simulation.get_n_registers()
        self.states_dim = 3 * self.actions_dim + self.registers_dim

    def get_frame(self):
        return (self.simulation.get_frame(self._cam_id) * 255).astype(np.uint8)

    def __del__(self):
        self.simulation.close()

    def restore(self, path):
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        self._critic_params = ckpt["critic"]
        self._actor_params = ckpt["actor"]

    def get_actions_returns(self):
        states = self.simulation.get_state()
        states = np.concatenate([states, [0, 0, 0, 0]])
        actions = self.agent._policy_network.apply(self._actor_params, states) # shape [N_ACTORS, ACTION_DIM]
        returns = self.agent._critic_network.apply(self._critic_params, states, actions) # shape [N_ACTORS]
        return actions, returns

    def get_return_plane(self, action, N=100):
        actions = get_action_plane(action, N=N)
        states = self.simulation.get_state()
        states = np.concatenate([states, [0, 0, 0, 0]])
        states = np.broadcast_to(states[np.newaxis, np.newaxis], (N, N, states.shape[-1]))
        return self.agent._critic_network.apply(self._critic_params, states, actions) # shape [N, N, 1]



class App(Frame):
    def __init__(self, gui_controller, master=None):
        self.gui_controller = gui_controller
        super().__init__(master)
        self.master.protocol("WM_DELETE_WINDOW", self.close)
        self.pack(side=TOP, fill=BOTH, expand=1)

        ########################################################################
        # LEFT
        ########################################################################
        left_pane = Frame(self)
        left_pane.pack(side=LEFT, fill=Y)
        self.robot_view = ImageZone(left_pane, 320, 240)
        self.robot_view.pack(fill=X)
        self.distance_matrix = PltFigure(left_pane, figsize=(3, 3))
        self.distance_matrix.pack(fill=X)
        self.distance_value = Label(left_pane)
        self.distance_value.config(font=(None, 25))
        self.distance_value.pack(side=TOP)

        self.qviewer = PltFigure(left_pane, figsize=(3, 3))
        self.qviewer.pack(fill=X)

        ########################################################################
        # RIGHT
        ########################################################################
        right_pane = Frame(self)
        right_pane.pack(side=RIGHT, fill=BOTH, expand=1)

        ########################################################################
        control_board = Frame(master=right_pane)
        control_board.pack(side=TOP, fill=X)

        joint_position_control = LabelFrame(padx=10, pady=10, text='Joint position control', master=control_board)
        joint_position_control.pack(side=TOP, fill=X)
        joint_intervals = [
            (-3, 3),
            (-3, 3),
            (-3, 3),
            (-2.1, 2.1),
            (-3, 3),
            (-2.1, 2.1),
            (-3, 3),
        ]
        self.sliders = tuple(
            Scale(joint_position_control, from_=from_, to=to, resolution=0.1, orient=HORIZONTAL)
            for from_, to in joint_intervals
        )
        for s in self.sliders: s.pack(side=TOP, fill=X)

        self.master.bind('a', lambda e: self.sliders[0].set(self.sliders[0].get() - 0.1))
        self.master.bind('z', lambda e: self.sliders[1].set(self.sliders[1].get() - 0.1))
        self.master.bind('e', lambda e: self.sliders[2].set(self.sliders[2].get() - 0.1))
        self.master.bind('r', lambda e: self.sliders[3].set(self.sliders[3].get() - 0.1))
        self.master.bind('t', lambda e: self.sliders[4].set(self.sliders[4].get() - 0.1))
        self.master.bind('y', lambda e: self.sliders[5].set(self.sliders[5].get() - 0.1))
        self.master.bind('u', lambda e: self.sliders[6].set(self.sliders[6].get() - 0.1))
        self.master.bind('q', lambda e: self.sliders[0].set(self.sliders[0].get() + 0.1))
        self.master.bind('s', lambda e: self.sliders[1].set(self.sliders[1].get() + 0.1))
        self.master.bind('d', lambda e: self.sliders[2].set(self.sliders[2].get() + 0.1))
        self.master.bind('f', lambda e: self.sliders[3].set(self.sliders[3].get() + 0.1))
        self.master.bind('g', lambda e: self.sliders[4].set(self.sliders[4].get() + 0.1))
        self.master.bind('h', lambda e: self.sliders[5].set(self.sliders[5].get() + 0.1))
        self.master.bind('j', lambda e: self.sliders[6].set(self.sliders[6].get() + 0.1))
        self.master.bind('<Return>', lambda e: self.move_arm())

        position_preview_button = Button(text='preview', padx=30, master=joint_position_control, command=self.move_arm)
        position_preview_button.pack(side=LEFT)
        position_compute_button = Button(text='compute', padx=30, master=joint_position_control, command=lambda: Thread(target=self.compute_all_frames).start())
        position_compute_button.pack(side=LEFT)
        position_compute_button = Button(text='step', padx=30, master=joint_position_control, command=self.step)
        position_compute_button.pack(side=LEFT)
        position_compute_button = Button(text='step (opt)', padx=30, master=joint_position_control, command=self.step_local_optimization)
        position_compute_button.pack(side=LEFT)

        self.progress_bar = ttk.Progressbar(joint_position_control, orient='horizontal', mode='determinate', length=200)
        self.progress_bar.pack(side=LEFT)

        register_state_control = LabelFrame(padx=10, pady=10, text='Register state control', master=control_board)
        register_state_control.pack(side=TOP, fill=X)
        self.register_variables = tuple(IntVar() for i in range(4))
        self.register_checkboxes = tuple(
            Checkbutton(register_state_control, text=f'{i+1}', variable=var, padx=30, command=self.update_register_states)
            for i, var in enumerate(self.register_variables)
        )
        for c in self.register_checkboxes: c.pack(side=LEFT)


        ########################################################################
        self.action_table = ScrollTable(right_pane)
        self.action_table.pack(side=TOP, fill=BOTH, expand=1)

        #define our columns
        self.action_table['columns'] = ('value', '1', '2', '3', '4', '5', '6', '7')
        # format our columns
        self.action_table.column("#0", width=0,  stretch=NO)
        for name in self.action_table['columns']:
            self.action_table.column(name, anchor=CENTER, width=80)
        #Create Headings
        self.action_table.heading("#0",text="",anchor=CENTER)
        for name in self.action_table['columns']:
            self.action_table.heading(name, text=name, anchor=CENTER)
        self.action_table.bind('<Motion>', self.table_motion)
        self.action_table.bind('<Leave>', self.update_robot_view)
        self.action_table.bind("<Double-Button-1>", self.table_double_click)

        ########################################################################
        self.target_position = None #tuple(s.get() for s in self.sliders)
        self.current_frame_shown = None
        self.arm_moved = True
        self.is_up_to_date = False
        self.state_tm1 = None
        self.register_states_tm1 = None
        self.action_tm1 = None
        self.reward_tm1 = None
        self.state_t = None
        self.action_t = None
        self.register_states_t = None
        self.new_environment_state()

    def close(self):
        self.gui_controller.simulation.close()
        self.master.destroy()

    def distance_matrix_hover(self, event):
        if event.inaxes == self.ax_distance_matrix:
            i, j = int(event.xdata), int(event.ydata)
            value = self.dmatrix[i, j]
            self.distance_value.config(text=f'({i}, {j})  -  {value:.3f}')
            if self.is_up_to_date:
                iframe = self.frames[i].astype(np.int32)
                jframe = self.frames[j].astype(np.int32)
                mean_frame = ((iframe + jframe) // 2).astype(np.uint8)
                self.robot_view(mean_frame)

    def qviewer_click(self, event):
        if event.inaxes == self.ax_qviewer:
            action = np.copy(self.qviewer_action)
            action[0], action[1] = event.xdata, event.ydata
            self.gui_controller.simulation.apply_action(action)
            self.arm_moved = True
            self.new_environment_state()

    def update_distance_matrix(self):
        self.dmatrix = np.sqrt(np.sum((self.actions[:, np.newaxis] - self.actions[np.newaxis]) ** 2, axis=-1))
        if not hasattr(self, 'ax_distance_matrix'):
            self.ax_distance_matrix = self.distance_matrix.fig.add_subplot(111)
            self.matshow = self.ax_distance_matrix.matshow(self.dmatrix)
            self.distance_matrix_cb = self.distance_matrix.fig.colorbar(self.matshow)
            self.distance_matrix.canvas.mpl_connect('motion_notify_event', self.distance_matrix_hover)
        else:
            self.matshow.set_data(self.dmatrix)
            self.distance_matrix_cb.mappable.set_clim(
                vmin=np.min(self.dmatrix),
                vmax=np.max(self.dmatrix),
            )
            self.distance_matrix.canvas.draw()

    def update_qviewer(self, event=None):
        selection = self.action_table.selection()
        if len(selection):
            actions = [tuple(
                float(v)
                for v in self.action_table.item(s, 'values')[1:]
            ) for s in selection]
            self.qviewer_action = np.mean(actions, axis=0)
        else:
            self.qviewer_action = self.action_t
        returns = self.gui_controller.get_return_plane(self.qviewer_action)
        if not hasattr(self, 'ax_qviewer'):
            self.ax_qviewer = self.qviewer.fig.add_subplot(111)
            returns = self.gui_controller.get_return_plane(self.qviewer_action)
            self.imshow = self.ax_qviewer.imshow(returns, extent=(-1, 1, -1, 1), origin='lower')
            self.marker, = self.ax_qviewer.plot(self.qviewer_action[0], self.qviewer_action[1], 'X', color='k')
            self.qviewer_cb = self.qviewer.fig.colorbar(self.imshow)
            self.action_table.bind('<<TreeviewSelect>>', self.update_qviewer)
            self.qviewer.canvas.mpl_connect('button_press_event', self.qviewer_click)
        else:
            self.imshow.set_data(returns)
            self.marker.set_xdata((self.qviewer_action[0],))
            self.marker.set_ydata((self.qviewer_action[1],))
            self.qviewer_cb.mappable.set_clim(
                vmin=np.min(self.returns) - 0.1,
                vmax=np.max(self.returns) + 0.1
            )
            self.qviewer.canvas.draw()

    def move_arm(self):
        target_position = tuple(s.get() for s in self.sliders)
        if target_position != self.target_position:
            self.target_position = target_position
            # self.gui_controller.simulation.set_joint_positions(target_position)
            self.gui_controller.simulation.set_control_loop_enabled(True)
            self.gui_controller.simulation.set_joint_target_positions(target_position)
            timeout = 10
            while not np.allclose(self.gui_controller.simulation.get_joint_positions(), target_position, rtol=1e-2, atol=1e-2) and timeout:
                self.gui_controller.simulation.step_sim()
                timeout -= 1
            self.gui_controller.simulation.set_control_loop_enabled(False)
            self.arm_moved = True
            self.new_environment_state()

    def step_local_optimization(self):
        # start_action = actions[np.argmax(returns)]
        state = self.state_t
        start_action = np.zeros(shape=7)
        bounds = Bounds([-1] * 7, [1] * 7)
        res = minimize(
            lambda action: -self.gui_controller.agent._critic_network.apply(self.gui_controller._critic_params, state, action),
            start_action,
            method='Nelder-Mead',
            bounds=bounds,
            options={'ftol': 0.01, 'disp':True},
        )
        print(res.x, res.fun)
        action = res.fun
        self.gui_controller.simulation.apply_action(np.array(action, dtype=np.float32))
        self.arm_moved = True
        self.new_environment_state()


    def step(self):
        action = self.action_t
        self.gui_controller.simulation.apply_action(np.array(action, dtype=np.float32))
        self.arm_moved = True
        self.new_environment_state()

    def table_double_click(self, event):
        row_str = self.action_table.identify_row(event.y)
        if row_str:
            row = int(row_str)
            action = self.actions[row]
            self.gui_controller.simulation.apply_action(np.array(action, dtype=np.float32))
            self.arm_moved = True
            self.new_environment_state()

    def table_motion(self, event):
        row_str = self.action_table.identify_row(event.y)
        if row_str and self.is_up_to_date:
            row = int(row_str)
            if row != self.current_frame_shown:
                self.robot_view(self.frames[row])
                self.current_frame_shown = row

    def compute_all_frames(self):
        self.gui_controller.simulation.set_reset_poses()
        register_states = self.gui_controller.simulation.get_stateful_objects_states()
        self.frames = []
        n = len(self.actions)
        for i, action in enumerate(self.actions):
            self.progress_bar['value'] = (100 * i) // n
            self.progress_bar.update()
            self.gui_controller.simulation.apply_action(action)
            self.frames.append(self.gui_controller.get_frame())
            self.gui_controller.simulation.reset_pose(register_states, [0, 0, 0, 0])
        self.progress_bar['value'] = 0
        self.progress_bar.update()
        self.is_up_to_date = True

    def new_environment_state(self):
        simulation = self.gui_controller.simulation
        self.is_up_to_date = False
        self.actions, self.returns = self.gui_controller.get_actions_returns()
        self.state_tm1 = self.state_t
        self.register_states_tm1 = self.register_states_t
        self.action_tm1 = self.action_t
        self.state_t = np.concatenate([simulation.get_state(), [0, 0, 0, 0]])
        self.register_states_t = simulation.get_stateful_objects_states()
        self.action_t = self.actions[np.argmax(self.returns)]
        if self.register_states_tm1 is not None:
            self.reward_tm1 = np.sum(self.register_states_tm1) - np.sum(self.register_states_t)
            print(f'{self.reward_tm1=}')
        else:
            self.reward_tm1 = None
        for var, state in zip(self.register_variables, self.register_states_t):
            var.set(state)
        joint_positions = simulation.get_joint_positions()
        # delete old data
        for item in self.action_table.get_children(): self.action_table.delete(item)
        # add new data
        for iid, (a, r) in enumerate(zip(self.actions, self.returns)):
            self.action_table.insert(
                parent='',
                index='end',
                iid=iid,
                text='',
                values=(r, ) + tuple(a),
            )
        self.update_robot_view()
        self.update_qviewer()
        self.update_distance_matrix()

    def update_robot_view(self, event=None):
        if self.arm_moved == True:
            self.frame_t = self.gui_controller.get_frame()
            self.arm_moved = False
        self.robot_view(self.frame_t)
        self.current_frame_shown = None

    def update_register_states(self):
        new = [var.get() for var in self.register_variables]
        self.gui_controller.simulation.set_stateful_objects_states(new)
        self.new_environment_state()


class ScrollTable(ttk.Treeview):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        scroll_x = Scrollbar(self, orient='horizontal')
        scroll_y = Scrollbar(self)
        # kwargs["xscrollcommand"] = scroll_x.set
        # kwargs["yscrollcommand"] = scroll_y.set
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.xview)
        scroll_y.config(command=self.yview)


class ImageZone(Canvas):
    def __init__(self, master, width, height):
        self.width, self.height = width, height
        super().__init__(master, width=width, height=height)
        Z = np.zeros((height, width, 3), dtype=np.uint8)
        data = f'P6 {width} {height} 255 '.encode() + Z.tobytes()
        self.image = PhotoImage(data=data, format='PPM')
        self.create_image(0, 0, anchor="nw", image=self.image)

    def __call__(self, array):
        height, width = array.shape[:2]
        assert height == self.height
        assert width == self.width
        data = f'P6 {width} {height} 255 '.encode() + array.tobytes()
        self.image = PhotoImage(data=data, format='PPM')
        self.create_image(0, 0, anchor="nw", image=self.image)


class PltFigure:
    def __init__(self, master, *args, with_toolbar=False, **kwargs):
        self.fig = Figure(**kwargs)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)  # A tk.DrawingArea.
        self.canvas.draw()
        if with_toolbar:
            self.toolbar = NavigationToolbar2Tk(self.canvas, master)
            self.toolbar.update()
        self.widget = self.canvas.get_tk_widget()
        # t = np.linspace(0, 10, 100)
        # self.fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    def pack(self, *args, **kwargs):
        self.widget.pack(*args, **kwargs)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    k = 1.15
    SQRT2 = 1.41421356237
    SAFETY = 2
    minmax_factor = 1.5
    dmin2 = 0.6
    dmax2 = dmin2 * minmax_factor
    dmin1 = SAFETY * SQRT2 * (dmax2)
    dmax1 = dmin1 * minmax_factor
    dmin0 = SAFETY * SQRT2 * (dmax1 + dmax2)
    dmax0 = dmin0 * minmax_factor

    dmin2 = 1.0
    hierarchization_config = (
        (50, dmin2, 100, 1 / k ** 2, 1 / k ** 2),
    )


    ws = Tk()
    ws.title('Contrastive DQN')
    ws.geometry('1000x500')

    gui_controller = GUIController(
        # "../checkpoints/latest_March18.ckpt",
        "../checkpoints/latest_March16_dmin2_0.3.ckpt",
        discount_factor=0.9,
        noise_magnitude_limit=2.0,
        hierarchization_config=hierarchization_config,
        hierarchization_coef=0.0001,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        action_dim=7,
    )
    app = App(gui_controller, ws)

    ws.mainloop()
