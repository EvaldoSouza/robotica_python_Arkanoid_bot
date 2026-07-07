import random
import numpy as np
import os
from typing import Tuple, Optional
from dataclasses import dataclass

from domain.models import Coordinate, FramePerception
from rl.storage_gateway import StorageGateway

StateVector = Tuple[int]

@dataclass(frozen=True)
class RlConfig:
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    q_dims: Tuple[int, ...] = (5, 3)
    paddle_y: float = 212.0
    left_wall: float = 16.0
    right_wall: float = 240.0
    hit_reward: float = 50.0


class StateDiscretizer:
    def discretize(self, perception: FramePerception) -> StateVector:
        if not self._has_required_entities(perception):
            return (2,) 

        rel_x = self._bin_relative_x(perception)
        return (rel_x,)

    def _has_required_entities(self, p: FramePerception) -> bool:
        return bool(p.paddle and p.ball)

    def _bin_relative_x(self, p: FramePerception) -> int:
        target_x = p.intercept_x if p.intercept_x != -1.0 else p.ball.x_pos # type: ignore
        diff_x = target_x - p.paddle.x_pos # type: ignore
        
        if diff_x < -20: return 0
        if diff_x < -5:  return 1
        if diff_x <= 5:  return 2
        if diff_x <= 20: return 3
        return 4


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
        if prev_velocity is None or not self._has_required_entities(perception):
            return 0.0, False

        is_hit = self._detect_hit(perception, prev_velocity)
        reward = self.config.hit_reward if is_hit else 0.0
        
        return reward, is_hit

    def _has_required_entities(self, p: FramePerception) -> bool:
        return bool(p.ball and p.paddle)

    def _detect_hit(self, p: FramePerception, prev_vel: Coordinate) -> bool:
        if p.velocity is None or p.ball is None:
            return False
            
        was_falling = prev_vel.y_pos > 0
        is_rising = p.velocity.y_pos < 0
        is_near_bottom = p.ball.y_pos > (self.config.paddle_y - 25)
        
        return was_falling and is_rising and is_near_bottom


class BrainArchive:
    def __init__(self, storage: StorageGateway, session_dir: str = "", filename: str = "arkanoid_brain.pkl") -> None:
        self.storage = storage
        self.filename = os.path.join(session_dir, filename) if session_dir else filename
        self.best_filename = os.path.join(session_dir, "arkanoid_best_brain.pkl") if session_dir else "arkanoid_best_brain.pkl"

    def load_brain(self, expected_shape: Tuple[int, ...]) -> Tuple[np.ndarray, float, int]:
        if not self.storage.exists(self.filename):
            return np.zeros(expected_shape), 1.0, 0
            
        return self._extract_payload(self.filename, expected_shape)

    def load_champion(self, expected_shape: Tuple[int, ...]) -> Tuple[np.ndarray, float, int]:
        if not self.storage.exists(self.best_filename):
            return np.zeros(expected_shape), 1.0, 0
            
        return self._extract_payload(self.best_filename, expected_shape)

    def save_brain(self, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        self._commit_to_storage(self.filename, q_table, epsilon, best_survival)

    def save_champion(self, q_table: np.ndarray, epsilon: float, best_survival: int) -> None:
        self._commit_to_storage(self.best_filename, q_table, epsilon, best_survival)

    def _extract_payload(
        self, filepath: str, expected_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, float, int]:
        
        payload = self.storage.read_pickle(filepath)
        if not payload:
            return np.zeros(expected_shape), 1.0, 0
            
        q_table = payload.get("q_table", np.zeros(expected_shape))
        
        if q_table.shape != expected_shape:
            raise ValueError(
                f"Storage shape mismatch. Offending value: {q_table.shape}. "
                f"Expected shape: {expected_shape}"
            )
            
        return q_table, payload.get("epsilon", 1.0), payload.get("best_survival", 0)

    def _commit_to_storage(
        self, filepath: str, q_table: np.ndarray, epsilon: float, best_survival: int
    ) -> None:
        self.storage.write_pickle(filepath, {
            "q_table": q_table.copy(), 
            "epsilon": epsilon, 
            "best_survival": best_survival
        })


class QLearningPolicy:
    def __init__(self, config: RlConfig, archive: BrainArchive, use_champion: bool = False) -> None:
        self.config = config
        self.archive = archive
        
        if use_champion:
            self.q_table, self.epsilon, self.best_survival = archive.load_champion(config.q_dims)
            self.epsilon = 0.0  # Force strictly greedy exploitation in showcase mode
        else:
            self.q_table, self.epsilon, self.best_survival = archive.load_brain(config.q_dims)

    def step_learning(
        self, curr_state: StateVector, reward: float, prev_state: StateVector, prev_action: int
    ) -> None:
        arr_action = prev_action - 1 
        state_action_tuple = prev_state + (arr_action,)
        
        old_q = self.q_table[state_action_tuple]
        max_future = np.max(self.q_table[curr_state])
        
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
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def clear_traces(self) -> None:
        pass

    def decay_exploration(self) -> float:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return self.epsilon


class ArkanoidBrain:
    def __init__(self, config: RlConfig, archive: BrainArchive, use_champion: bool = False) -> None:
        self.config = config
        self.archive = archive
        self.discretizer = StateDiscretizer()
        self.shaper = RewardShaper(config)
        self.policy = QLearningPolicy(config, archive, use_champion)
        self.prev_velocity: Optional[Coordinate] = None

    def calculate_reward(
        self, perception: FramePerception, missing_frames: int, prev_action: int, prev_prev_action: int
    ) -> float:
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
        terminal_delta = penalty - old_q
        self.policy.q_table[state_action_tuple] = old_q + (self.config.alpha * terminal_delta)

    def decay_exploration_rate(self, current_survival_frames: int = 0, current_score: int = 0) -> float:
        is_new_best = current_survival_frames > self.policy.best_survival
        
        if is_new_best:
            self.policy.best_survival = current_survival_frames
            self.archive.save_champion(self.policy.q_table, self.policy.epsilon, self.policy.best_survival)
            
        self.archive.save_brain(self.policy.q_table, self.policy.epsilon, self.policy.best_survival)
        return self.policy.decay_exploration()