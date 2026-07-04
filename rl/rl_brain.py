import os
import random
import numpy as np
import pickle
from typing import Tuple, Optional
from dataclasses import dataclass

from domain.models import Coordinate, FramePerception

StateVector = Tuple[int, int, int, int]

@dataclass(frozen=True)
class RlConfig:
    alpha: float = 0.2
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    lambda_trace: float = 0.9
    q_dims: Tuple[int, ...] = (5, 3, 2, 2, 3)
    paddle_y: float = 212.0
    left_wall: float = 16.0
    right_wall: float = 240.0


class StateDiscretizer:
    """
    Converts continuous geometric coordinates into a discrete 4D state vector.
    Indices are 0-based for strict numpy compatibility.
    """
    
    def discretize(self, perception: FramePerception) -> Tuple[int, int, int, int]:
        if not self._is_valid_perception(perception):
            return (2, 0, 0, 0) # Fallback: Center, High, Left, Up

        rel_x = self._bin_relative_x(perception)
        ball_y = self._bin_ball_y(perception.ball.y_pos) # type: ignore
        dir_x, dir_y = self._bin_velocity(perception.velocity) # type: ignore
        
        return (rel_x, ball_y, dir_x, dir_y)

    def _is_valid_perception(self, p: FramePerception) -> bool:
        has_objects = p.paddle is not None and p.ball is not None
        has_velocity = p.velocity is not None
        return has_objects and has_velocity

    def _bin_relative_x(self, p: FramePerception) -> int:
        target_x = p.intercept_x if p.intercept_x != -1.0 else p.ball.x_pos # type: ignore
        diff_x = target_x - p.paddle.x_pos # type: ignore
        
        if diff_x < -20: return 0
        if diff_x < -5: return 1
        if diff_x <= 5: return 2
        if diff_x <= 20: return 3
        return 4

    def _bin_ball_y(self, y_pos: float) -> int:
        if y_pos < 100: return 0
        if y_pos < 180: return 1
        return 2

    def _bin_velocity(self, vel: Coordinate) -> Tuple[int, int]:
        dx = 0 if vel.x_pos < 0 else 1
        dy = 0 if vel.y_pos < 0 else 1
        return dx, dy



class RewardShaper:
    """
    Calculates RL rewards based purely on visual physics.
    Proves mathematically to the agent that prolonging the episode is inherently good.
    """
    def __init__(self, config: RlConfig) -> None:
        self.config = config

    def calculate(
        self, 
        perception: FramePerception, 
        prev_action: int, 
        prev_prev_action: int, 
        prev_velocity: Optional[Coordinate]
    ) -> float:
        if not self._is_valid(perception, prev_velocity):
            return 0.0

        reward = 0.1 # Base survival reward
        
        reward = self._apply_hit_reward(reward, perception, prev_velocity) # type: ignore
        reward = self._apply_jitter_penalty(reward, prev_action, prev_prev_action)
        reward = self._apply_wall_penalty(reward, perception.paddle.x_pos, prev_action) # type: ignore
        reward = self._apply_tracking_reward(reward, perception)
        
        return reward

    def _is_valid(self, p: FramePerception, prev_vel: Optional[Coordinate]) -> bool:
        return p.ball is not None and p.paddle is not None and prev_vel is not None

    def _apply_hit_reward(
        self, reward: float, p: FramePerception, prev_vel: Coordinate
    ) -> float:
        if p.velocity is None:
            return reward
            
        was_falling = prev_vel.y_pos > 0
        is_rising = p.velocity.y_pos < 0
        is_near_bottom = p.ball.y_pos > (self.config.paddle_y - 25) # type: ignore
        
        if was_falling and is_rising and is_near_bottom:
            return reward + 50.0
        return reward

    def _apply_jitter_penalty(
        self, reward: float, action1: int, action2: int
    ) -> float:
        # Actions 1 & 2 are Left/Right in the orchestrator
        if (action1 == 1 and action2 == 2) or (action1 == 2 and action2 == 1):
            return reward - 1.0
        return reward

    def _apply_wall_penalty(
        self, reward: float, paddle_x: float, prev_action: int
    ) -> float:
        is_grinding_left = (paddle_x < self.config.left_wall + 15) and (prev_action == 1)
        is_grinding_right = (paddle_x > self.config.right_wall - 15) and (prev_action == 2)
        
        if is_grinding_left or is_grinding_right:
            return reward - 1.0
        return reward

    def _apply_tracking_reward(
        self, reward: float, p: FramePerception
    ) -> float:
        target_x = p.intercept_x if p.intercept_x != -1.0 else p.ball.x_pos # type: ignore
        if abs(target_x - p.paddle.x_pos) <= 5.0: # type: ignore
            return reward + 0.05
        return reward


class BrainStorage:
    """Manages disk persistence for the Q-Table and training checkpoints."""
    
    def __init__(self, filename: str = "arkanoid_brain.pkl") -> None:
        self.filename = filename
        self.best_filename = "arkanoid_best_brain.pkl"

    def load_brain(self, expected_shape: Tuple[int, ...]) -> Tuple[np.ndarray, float, int]:
        if not os.path.exists(self.filename):
            return np.zeros(expected_shape), 1.0, 0
            
        with open(self.filename, "rb") as f:
            data = pickle.load(f)
            
        q_table = data.get("q_table", np.zeros(expected_shape))
        epsilon = data.get("epsilon", 1.0)
        best_survival = data.get("best_survival", 0)
        
        if q_table.shape != expected_shape:
            raise ValueError(
                f"Brain mismatch. Expected {expected_shape}, got {q_table.shape}"
            )
            
        return q_table, epsilon, best_survival

    def save_brain(self, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        self._write_to_disk(self.filename, q_table, epsilon, best_survival)

    def save_champion(self, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        self._write_to_disk(self.best_filename, q_table, epsilon, best_survival)

    def _write_to_disk(self, filepath: str, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({"q_table": q_table, "epsilon": epsilon, "best_survival": best_survival}, f)



class TDLambdaPolicy:
    """
    Implements the TD(Lambda) Bellman equation with Watkins's replacing traces.
    """
    def __init__(self, config: RlConfig, storage: BrainStorage) -> None:
        self.config = config
        self.storage = storage
        self.q_table, self.epsilon, self.best_survival_from_disk = storage.load_brain(config.q_dims)
        self.e_table = np.zeros(config.q_dims)

    def step_learning(
        self, curr_state: StateVector, reward: float, prev_state: StateVector, prev_action: int
    ) -> None:
        # Actions from the orchestrator are 1, 2, 3. Array indices must be 0, 1, 2.
        arr_action = prev_action - 1 
        
        old_q = self.q_table[prev_state + (arr_action,)]
        max_future = np.max(self.q_table[curr_state])
        
        delta = reward + self.config.gamma * max_future - old_q
        self.e_table[prev_state + (arr_action,)] = 1.0 # Replacing trace
        
        self.q_table += self.config.alpha * delta * self.e_table
        self.e_table *= (self.config.gamma * self.config.lambda_trace)

    def select_action(self, state: StateVector) -> Tuple[int, bool]:
        if random.random() < self.epsilon:
            return random.randint(0, 2), True
            
        action_idx = int(np.argmax(self.q_table[state]))
        return action_idx, False

    def clear_traces(self) -> None:
        self.e_table.fill(0.0)

    def decay_exploration(self) -> float:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return self.epsilon



class ArkanoidBrain:
    """
    Facade for the RL components. Matches the BrainInterface expected by main.py.
    """
    def __init__(self) -> None:
        self.config = RlConfig()
        self.storage = BrainStorage()
        self.discretizer = StateDiscretizer()
        self.shaper = RewardShaper(self.config)
        self.policy = TDLambdaPolicy(self.config, self.storage)
        
        self.best_survival = self.policy.best_survival_from_disk
        self.prev_velocity: Optional[Coordinate] = None

    def calculate_reward(self, perception: FramePerception, missing_frames: int, prev_action: int, prev_prev_action: int) -> float:
        if missing_frames >= 15:
            return -100.0
            
        reward = self.shaper.calculate(perception, prev_action, prev_prev_action, self.prev_velocity)
        self.prev_velocity = perception.velocity
        return reward

    def decide_optimal_action(self, state_tuple: StateVector) -> int:
        action_arr_idx = int(np.argmax(self.policy.q_table[state_tuple]))
        return action_arr_idx + 1 # Convert array index 0-2 back to orchestrator action 1-3

    def decide_exploratory_action(
        self, state: Tuple, old_state: Optional[Tuple], old_action: int, reward: float
    ) -> int:
        if old_state is not None and old_action > 0:
            self.policy.step_learning(state, reward, old_state, old_action)
            
        arr_action, is_exploratory = self.policy.select_action(state)
        
        if is_exploratory:
            self.policy.clear_traces()
            
        return arr_action + 1 

    def apply_terminal_penalty(self, old_state: Tuple, old_action: int, penalty: float) -> None:
        if old_action <= 0:
            return
            
        arr_action = old_action - 1
        old_q = self.policy.q_table[old_state + (arr_action,)]
        terminal_delta = penalty - old_q
        self.policy.q_table += (self.config.alpha * terminal_delta * self.policy.e_table)
        
        self.policy.clear_traces()

    def decay_exploration_rate(self, current_survival_frames: int = 0) -> float:
        if current_survival_frames > self.best_survival and current_survival_frames > 100:
            self.best_survival = current_survival_frames
            self.storage.save_champion(self.policy.q_table, self.policy.epsilon, self.best_survival)
            
        self.storage.save_brain(self.policy.q_table, self.policy.epsilon, self.best_survival)
        return self.policy.decay_exploration()