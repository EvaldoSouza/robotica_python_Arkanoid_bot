import os
import random
import pickle
import numpy as np
from typing import Tuple, Optional, Any
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
    hit_reward: float = 50.0
    tracking_reward: float = 0.5
    jitter_penalty: float = 1.0
    wall_grind_penalty: float = 1.0


class StateDiscretizer:
    def discretize(self, perception: FramePerception) -> StateVector:
        if not self._is_valid_perception(perception):
            return (2, 0, 0, 0)

        rel_x = self._bin_relative_x(perception)
        ball_y = self._bin_ball_y(perception.ball.y_pos)  # type: ignore
        dir_x, dir_y = self._bin_velocity(perception.velocity)  # type: ignore
        
        return (rel_x, ball_y, dir_x, dir_y)

    def _is_valid_perception(self, p: FramePerception) -> bool:
        return bool(p.paddle and p.ball and p.velocity)

    def _bin_relative_x(self, p: FramePerception) -> int:
        target_x = p.intercept_x if p.intercept_x != -1.0 else p.ball.x_pos  # type: ignore
        diff_x = target_x - p.paddle.x_pos  # type: ignore
        
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
    def __init__(self, config: RlConfig) -> None:
        self.config = config

    def calculate(
        self, 
        perception: FramePerception, 
        prev_action: int, 
        prev_prev_action: int, 
        prev_velocity: Optional[Coordinate]
    ) -> Tuple[float, bool]:
        """Calculates visual physics reward and detects if a paddle hit occurred."""
        if not self._is_valid(perception, prev_velocity):
            return 0.0, False

        reward = 0.1
        is_hit = self._detect_hit(perception, prev_velocity) # type: ignore
        
        if is_hit:
            reward += self.config.hit_reward

        reward = self._apply_jitter_penalty(reward, prev_action, prev_prev_action)
        reward = self._apply_wall_penalty(reward, perception.paddle.x_pos, prev_action) # type: ignore
        reward = self._apply_tracking_reward(reward, perception)
        
        return reward, is_hit

    def _is_valid(self, p: FramePerception, prev_vel: Optional[Coordinate]) -> bool:
        return bool(p.ball and p.paddle and prev_vel)

    def _detect_hit(self, p: FramePerception, prev_vel: Coordinate) -> bool:
        if p.velocity is None or p.ball is None:
            return False
            
        was_falling = prev_vel.y_pos > 0
        is_rising = p.velocity.y_pos < 0
        is_near_bottom = p.ball.y_pos > (self.config.paddle_y - 25)
        
        return was_falling and is_rising and is_near_bottom

    def _apply_jitter_penalty(self, reward: float, action1: int, action2: int) -> float:
        if (action1 == 1 and action2 == 2) or (action1 == 2 and action2 == 1):
            return reward - self.config.jitter_penalty
        return reward

    def _apply_wall_penalty(self, reward: float, paddle_x: float, prev_action: int) -> float:
        is_grinding_left = (paddle_x < self.config.left_wall + 15) and (prev_action == 1)
        is_grinding_right = (paddle_x > self.config.right_wall - 15) and (prev_action == 2)
        
        if is_grinding_left or is_grinding_right:
            return reward - self.config.wall_grind_penalty
        return reward

    def _apply_tracking_reward(self, reward: float, p: FramePerception) -> float:
        target_x = p.intercept_x if p.intercept_x != -1.0 else p.ball.x_pos  # type: ignore
        if abs(target_x - p.paddle.x_pos) <= 5.0:  # type: ignore
            return reward + self.config.tracking_reward
        return reward


class BrainArchive:
    def __init__(self, filename: str = "arkanoid_brain.pkl") -> None:
        self.filename = filename
        self.best_filename = "arkanoid_best_brain.pkl"

    def load_brain(self, expected_shape: Tuple[int, ...]) -> Tuple[np.ndarray, float, int]:
        if not os.path.exists(self.filename):
            return np.zeros(expected_shape), 1.0, 0
            
        return self._read_file(self.filename, expected_shape)

    def load_champion(self, expected_shape: Tuple[int, ...]) -> np.ndarray:
        if not os.path.exists(self.best_filename):
            return np.zeros(expected_shape)
            
        q_table, _, _ = self._read_file(self.best_filename, expected_shape)
        return q_table

    def save_brain(self, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        self._write_to_disk(self.filename, q_table, epsilon, best_survival)

    def save_champion(self, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        self._write_to_disk(self.best_filename, q_table, epsilon, best_survival)

    def _read_file(self, filepath: str, expected_shape: Tuple[int, ...]) -> Tuple[np.ndarray, float, int]:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        q_table = data.get("q_table", np.zeros(expected_shape))
        
        if q_table.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch. Offending value: {q_table.shape}. Expected shape: {expected_shape}"
            )
            
        return q_table, data.get("epsilon", 1.0), data.get("best_survival", 0)

    def _write_to_disk(self, filepath: str, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": q_table.copy(), 
                "epsilon": epsilon, 
                "best_survival": best_survival
            }, f)


class TDLambdaPolicy:
    def __init__(self, config: RlConfig, archive: BrainArchive) -> None:
        self.config = config
        self.archive = archive
        self.q_table, self.epsilon, self.best_survival = archive.load_brain(config.q_dims)
        self.e_table = np.zeros(config.q_dims)

    def step_learning(
        self, curr_state: StateVector, reward: float, prev_state: StateVector, prev_action: int
    ) -> None:
        arr_action = prev_action - 1 
        state_action_tuple = prev_state + (arr_action,)
        
        old_q = self.q_table[state_action_tuple]
        max_future = np.max(self.q_table[curr_state])
        
        delta = reward + self.config.gamma * max_future - old_q
        self.e_table[state_action_tuple] = 1.0 
        
        self.q_table += self.config.alpha * delta * self.e_table
        self.e_table *= (self.config.gamma * self.config.lambda_trace)

    def select_action(self, state: StateVector) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 2)
            
        return int(np.argmax(self.q_table[state]))

    def select_greedy_action(self, state: StateVector) -> int:
        return int(np.argmax(self.q_table[state]))

    def clear_traces(self) -> None:
        self.e_table.fill(0.0)

    def decay_exploration(self) -> float:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return self.epsilon


class ArkanoidBrain:
    def __init__(self, config: RlConfig, archive: BrainArchive) -> None:
        self.config = config
        self.archive = archive
        self.discretizer = StateDiscretizer()
        self.shaper = RewardShaper(config)
        self.policy = TDLambdaPolicy(config, archive)
        
        self.prev_velocity: Optional[Coordinate] = None
        self._was_hit_this_frame: bool = False

    def calculate_reward(self, perception: FramePerception, missing_frames: int, prev_action: int, prev_prev_action: int) -> float:
        if missing_frames >= 15:
            return -100.0
            
        reward, self._was_hit_this_frame = self.shaper.calculate(
            perception, prev_action, prev_prev_action, self.prev_velocity
        )
        self.prev_velocity = perception.velocity
        return reward

    def decide_optimal_action(self, state_tuple: StateVector) -> int:
        """Used for evaluating the true strength of the champion brain."""
        action_arr_idx = self.policy.select_greedy_action(state_tuple)
        return action_arr_idx + 1 

    def decide_exploratory_action(
        self, state: StateVector, old_state: Optional[StateVector], old_action: int, reward: float
    ) -> int:
        if old_state is not None and old_action > 0:
            self.policy.step_learning(state, reward, old_state, old_action)
            
        # Domain-driven trace cutoff: Clear traces only AFTER a successful volley hit.
        if self._was_hit_this_frame:
            self.policy.clear_traces()
            
        arr_action = self.policy.select_action(state)
        return arr_action + 1 

    def apply_terminal_penalty(self, old_state: StateVector, old_action: int, penalty: float) -> None:
        if old_action <= 0:
            return
            
        arr_action = old_action - 1
        state_action_tuple = old_state + (arr_action,)
        
        old_q = self.policy.q_table[state_action_tuple]
        terminal_delta = penalty - old_q
        self.policy.q_table += (self.config.alpha * terminal_delta * self.policy.e_table)
        
        self.policy.clear_traces()

    def decay_exploration_rate(self, current_survival_frames: int = 0, current_score: int = 0) -> float:
        # Improved champion saving: Require actual points, not just survival frames.
        is_new_best = current_survival_frames > self.policy.best_survival and current_score > 0
        
        if is_new_best:
            self.policy.best_survival = current_survival_frames
            self.archive.save_champion(self.policy.q_table, self.policy.epsilon, self.policy.best_survival)
            
        self.archive.save_brain(self.policy.q_table, self.policy.epsilon, self.policy.best_survival)
        return self.policy.decay_exploration()