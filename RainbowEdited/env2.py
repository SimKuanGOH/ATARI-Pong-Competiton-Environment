# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch
import numpy as np
import time

class Env():
  def __init__(self, env):
    self.device = "cpu"
#     self.ale = atari_py.ALEInterface()
#     self.ale.setInt('random_seed', args.seed)
#     self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
#     self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
#     self.ale.setInt('frame_skip', 0)
#     self.ale.setBool('color_averaging', False)
#     self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = np.array([ 0,  1,  3,  4, 11, 12])#self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
#     self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = 4#args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=4)
    self.training = False  # Consistent with model training mode
    self.screen = None
    self.env = env
    self.done = None
    self.info = None
    self.players = env.players

  def interact(self, act0):
    
    act = np.array([1,0,0,0,  0,0,   0,0,  0,0,0,0, 0,0,0,1])
    
    if act0 == 2 or act0 ==4:
        act[4] = 1
    if act0 == 3 or act0 ==5:
        act[5] = 1
        
    act1 = act
    
    obs, rew, done, info = self.env.step(act1)
    self.done = done
    self.screen = obs
    self.info = info
    return np.array(rew)

  def interact_2P(self, act01, act02):
    
    act = np.array([1,0,0,0,  0,0,   0,0,  0,0,0,0, 0,0,0,1])
    
    if act01 == 2 or act01 ==4:
        act[4] = 1
    if act01 == 3 or act01 ==5:
        act[5] = 1
    if act02 == 2 or act02 ==4:
        act[6] = 1
    if act02 == 3 or act02 ==5:
        act[7] = 1
        
    act1 = act
    
    obs, rew, done, info = self.env.step(act1)
    self.done = done
    self.screen = obs
    self.info = info
    return np.array(rew)

  def _get_state(self):
    state = cv2.resize(cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    # if self.done:
    #   self.done = False  # Reset flag
    #   self.interact(0)  # Use a no-op after loss of life
    # else:
    if True:
      # Reset internals
      self._reset_buffer()
      self.env.reset()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.interact(0)  # Assumes raw action 0 is always no-op
        if self.done:
          self.env.reset()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
#     self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    if self.players == 1:
        reward, done = 0, False
    if self.players == 2:
        reward, done = np.array([0.0,0.0]), False    
        
    for t in range(4):
      reward += self.interact(action)
    
    
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.done
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    # if self.training:
    #   lives = self.ale.lives()
    #   if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
    #     self.life_termination = not done  # Only set flag when not truly done
    #     done = True
    #   self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done, self.info


  def step_2P(self, action1, action2):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    if self.players == 1:
        reward, done = 0, False
    if self.players == 2:
        reward, done = np.array([0.0,0.0]), False    
        
    for t in range(4):
      reward += self.interact_2P(action1, action2)
    
    
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.done
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    # if self.training:
    #   lives = self.ale.lives()
    #   if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
    #     self.life_termination = not done  # Only set flag when not truly done
    #     done = True
    #   self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done, self.info

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    # cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    # cv2.waitKey(1)
    self.env.render()
    time.sleep(0.01)

  def close(self):
    cv2.destroyAllWindows()
