import os
import random
import pickle
import numpy as np
from typing import Tuple, Optional, Any
from dataclasses import dataclass

from domain.models import Coordinate, FramePerception

# Simplified to a 1D state representation: (relative_x_bin,)
StateVector = Tuple[int]


@dataclass(frozen=True)
class RlConfig:
    alpha: float = 0.2
    gamma: float = 0.95  # Slightly lowered to focus on near-term interception
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    q_dims: Tuple[int, ...] = (5, 3)  # 5 States (Relative X alignment) x 3 Actions
    paddle_y: float = 212.0
    left_wall: float = 16.0
    right_wall: float = 240.0
    hit_reward: float = 50.0


class StateDiscretizer:
    def discretize(self, perception: FramePerception) -> StateVector:
        if not self._is_valid_perception(perception):
            return (2,)  # Default to 'Centered' state if data is missing

        rel_x = self._bin_relative_x(perception)
        return (rel_x,)

    def _is_valid_perception(self, p: FramePerception) -> bool:
        return bool(p.paddle and p.ball)

    def _bin_relative_x(self, p: FramePerception) -> int:
        # Use your custom physics intercept projection if active, otherwise use raw ball position
        target_x = p.intercept_x if p.intercept_x != -1.0 else p.ball.x_pos  # type: ignore
        diff_x = target_x - p.paddle.x_pos  # type: ignore
        
        # Discretize the distance between the paddle center and the target landing point
        if diff_x < -20: return 0   # Far Left
        if diff_x < -5:  return 1   # Near Left
        if diff_x <= 5:  return 2   # Centered
        if diff_x <= 20: return 3   # Near Right
        return 4                    # Far Right


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
        """Calculates sparse hit rewards based purely on ball bounce detection."""
        if not self._is_valid(perception, prev_velocity):
            return 0.0, False

        is_hit = self._detect_hit(perception, prev_velocity) # type: ignore
        reward = self.config.hit_reward if is_hit else 0.0
        
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


class QLearningPolicy:
    def __init__(self, config: RlConfig, archive: BrainArchive) -> None:
        self.config = config
        self.archive = archive
        self.q_table, self.epsilon, self.best_survival = archive.load_brain(config.q_dims)

    def step_learning(
        self, curr_state: StateVector, reward: float, prev_state: StateVector, prev_action: int
    ) -> None:
        # Convert engine action (1, 2, 3) to 0-indexed array space (0, 1, 2)
        arr_action = prev_action - 1 
        state_action_tuple = prev_state + (arr_action,)
        
        old_q = self.q_table[state_action_tuple]
        max_future = np.max(self.q_table[curr_state])
        
        # Standard Vanilla Q-Learning Update Formula
        self.q_table[state_action_tuple] = old_q + self.config.alpha * (
            reward + self.config.gamma * max_future - old_q
        )

    def select_action(self, state: StateVector) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 2)
            
        return self.select_greedy_action(state)

    def select_greedy_action(self, state: StateVector) -> int:
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        
        # Find all actions that share the highest Q-value
        best_actions = np.where(q_values == max_q)[0]
        
        # Randomly choose one of the tied actions
        return int(np.random.choice(best_actions))

    def clear_traces(self) -> None:
        """Kept to preserve interface compatibility with external systems."""
        pass

    def decay_exploration(self) -> float:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return self.epsilon


class ArkanoidBrain:
    def __init__(self, config: RlConfig, archive: BrainArchive) -> None:
        self.config = config
        self.archive = archive
        self.discretizer = StateDiscretizer()
        self.shaper = RewardShaper(config)
        self.policy = QLearningPolicy(config, archive)
        
        self.prev_velocity: Optional[Coordinate] = None

    def calculate_reward(self, perception: FramePerception, missing_frames: int, prev_action: int, prev_prev_action: int) -> float:
        # Terminal condition: Agent failed to catch the ball
        if missing_frames >= 15:
            return -100.0
            
        reward, _ = self.shaper.calculate(
            perception, prev_action, prev_prev_action, self.prev_velocity
        )
        self.prev_velocity = perception.velocity
        return reward

    def decide_optimal_action(self, state_tuple: StateVector) -> int:
        action_arr_idx = self.policy.select_greedy_action(state_tuple)
        return action_arr_idx + 1 

    def decide_exploratory_action(
        self, state: StateVector, old_state: Optional[StateVector], old_action: int, reward: float
    ) -> int:
        if old_state is not None and old_action > 0:
            self.policy.step_learning(state, reward, old_state, old_action)
            
        arr_action = self.policy.select_action(state)
        return arr_action + 1 

    def apply_terminal_penalty(self, old_state: StateVector, old_action: int, penalty: float) -> None:
        if old_action <= 0:
            return
            
        arr_action = old_action - 1
        state_action_tuple = old_state + (arr_action,)
        
        old_q = self.policy.q_table[state_action_tuple]
        
        # In a terminal state, there is no future value max_Q(s', a'), so it evaluates to 0
        terminal_delta = penalty - old_q
        self.policy.q_table[state_action_tuple] = old_q + (self.config.alpha * terminal_delta)

    def decay_exploration_rate(self, current_survival_frames: int = 0, current_score: int = 0) -> float:
        is_new_best = current_survival_frames > self.policy.best_survival
        
        if is_new_best:
            self.policy.best_survival = current_survival_frames
            self.archive.save_champion(self.policy.q_table, self.policy.epsilon, self.policy.best_survival)
            
        self.archive.save_brain(self.policy.q_table, self.policy.epsilon, self.policy.best_survival)
        return self.policy.decay_exploration()