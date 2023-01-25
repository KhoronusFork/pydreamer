import os

import gymnasium as gym
import gymnasium.spaces
import numpy as np

from dm_control import _render
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import viewer

_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_MAX_FRONTBUFFER_SIZE = 2048
_DEFAULT_WINDOW_TITLE = 'Rendering Observer'


class Observer:
  """A stripped down 3D renderer for Mujoco environments.
  Attributes:
    camera_config: A dict.  The camera configuration for the observer.
  """

  def __init__(self, env, width, height, name):
    """Observer constructor.
    Args:
      env: The environment.
      width: Window width, in pixels.
      height: Window height, in pixels.
      name: Window name.
    """
    self._env = env
    self._viewport = renderer.Viewport(width, height)
    self._viewer = None
    self._camera_config = {
        'lookat': None,
        'distance': None,
        'azimuth': None,
        'elevation': None
    }
    self._camera_config_dirty = False

    self._render_surface = _render.Renderer(
        max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
    self._renderer = renderer.NullRenderer()
    self._window = gui.RenderWindow(width, height, name)

  @classmethod
  def build(cls, env, height=_DEFAULT_HEIGHT, width=_DEFAULT_WIDTH,
            name=_DEFAULT_WINDOW_TITLE):
    """Returns a Observer with a default platform.
    Args:
      env: The environment.
      height: Window height, in pixels.
      width: Window width, in pixels.
      name: Window name.
    Returns:
      Newly constructor Observer.
    """
    return cls(env, width, height, name)

  def _apply_camera_config(self):
    for key, value in self._camera_config.items():
      if value is not None:
        if key == 'lookat':  # special case since we can't just set this attr.
          self._viewer.camera.settings.lookat[:] = self._camera_config['lookat']
        else:
          setattr(self._viewer.camera.settings, key, value)

    self._camera_config_dirty = False

  @property
  def camera_config(self):
    """Retrieves the current camera configuration."""
    if self._viewer:
      for key, value in self._camera_config.items():
        self._camera_config[key] = getattr(self._viewer.camera.settings, key,
                                           value)
    return self._camera_config

  @camera_config.setter
  def camera_config(self, camera_config):
    for key, value in camera_config.items():
      if key not in self._camera_config:
        raise ValueError(('Key {} is not a valid key in camera_config. '
                          'Valid keys are: {}').format(
                              key, list(camera_config.keys())))
      self._camera_config[key] = value
    self._camera_config_dirty = True

  def begin_episode(self, *unused_args, **unused_kwargs):
    """Notifies the observer that a new episode is about to begin.
    Args:
      *unused_args: ignored.
      **unused_kwargs: ignored.
    """
    if not self._viewer:
      self._viewer = viewer.Viewer(
          self._viewport, self._window.mouse, self._window.keyboard)
    if self._viewer:
      self._renderer = renderer.OffScreenRenderer(
          self._env.physics.model, self._render_surface)
      self._viewer.initialize(self._env.physics, self._renderer, False)

  def end_episode(self, *unused_args, **unused_kwargs):
    """Notifies the observer that an episode has ended.
    Args:
      *unused_args: ignored.
      **unused_kwargs: ignored.
    """
    if self._viewer:
      self._viewer.deinitialize()

  def _render(self):
    self._viewport.set_size(*self._window.shape)
    self._viewer.render()
    return self._renderer.pixels

  def step(self, *unused_args, **unused_kwargs):
    """Notifies the observer that an agent has taken a step.
    Args:
      *unused_args: ignored.
      **unused_kwargs: ignored.
    """
    if self._viewer:
      if self._camera_config_dirty:
        self._apply_camera_config()
    self._window.update(self._render)

class DMC(gym.Env):

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        # os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if domain == 'manip':
            from dm_control import manipulation
            self._env = manipulation.load(task + '_vision')
        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite
            self._env = suite.load(domain, task)

        # Launch the viewer application.
        # https://github.com/deepmind/dm_control/blob/main/dm_control/viewer/README.md
        print('viewer')
        self.observer = Observer(self._env, 640, 480, 'viewobs')
        self.observer.begin_episode()
        #initial_camera_cfg = {
        #        'distance': 1.0,
        #        'azimuth': 30.0,
        #        'elevation': -45.0,
        #        'lookat': [0.0, 0.1, 0.2],
        #    }
        #self.observer.camera_config = initial_camera_cfg
        #self.observer._viewer.camera.settings.lookat = np.zeros(3)
        print('viewer done')

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return action

    def step(self, action):
        assert np.isfinite(action).all(), action  # type: ignore
        reward = 0
        time_step = None
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        assert time_step is not None
        obs = self.observation(time_step)
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        self.observer.step()
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self.observation(time_step)
        return obs

    def observation(self, time_step):
        obs = dict(time_step.observation)
        obs = {k: v for k, v in obs.items() if k not in self._ignored_keys}
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)  # type: ignore
