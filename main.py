import sys
import argparse
import logging
import json
import os
from enum import Enum
from typing import Optional
from dataclasses import dataclass

import numpy as np

# Domain Imports
from emulator.nes_environment import NesEnvironment
from vision.vision_pipeline import VisionPipeline, VisionConfig, PhysicsEnvironment
from rl.rl_brain import ArkanoidBrain, RlConfig, BrainArchive
from display.agent_dashboard import DashboardManager
from domain.models import FramePerception, TelemetryHistory

# Configure user-facing CLI output (Plain Text)
cli_logger = logging.getLogger("arkanoid_cli")
cli_logger.setLevel(logging.INFO)
cli_handler = logging.StreamHandler(sys.stdout)
cli_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
cli_logger.addHandler(cli_handler)

# Configure debugging/observability output (Structured JSON)
debug_logger = logging.getLogger("arkanoid_debug")
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler("arkanoid_debug.log")
debug_handler.setFormatter(logging.Formatter("%(message)s"))
debug_logger.addHandler(debug_handler)


class ExecutionMode(Enum):
    SHOWCASE = "showcase"
    TRAIN_UI = "train_ui"
    TRAIN_HEADLESS = "train_headless"


@dataclass
class EpisodeTelemetry:
    steps_survived: int = 0
    accumulated_reward: float = 0.0
    paddle_hits: int = 0
    blocks_destroyed: int = 0
    action_jitters: int = 0


class TelemetryTracker:
    """
    Encapsulates all domain rules for tracking game state, scoring,
    and detecting specific agent behaviors like jitter or paddle hits.
    """
    def __init__(self) -> None:
        self.stats = EpisodeTelemetry()
        self.history = TelemetryHistory([], [], [], [], [], [], [])
        self.ball_absence_counter = 0
        self.prev_action = 0
        self.prev_prev_action = 0
        self.prev_perception: Optional[FramePerception] = None
        self.block_buffer: list[int] = []

    def reset_episode(self) -> None:
        self.stats = EpisodeTelemetry()
        self.prev_perception = None
        self.prev_action = 0
        self.prev_prev_action = 0
        self.ball_absence_counter = 0
        self.block_buffer.clear()

    def reset_debounce_buffer(self, initial_count: int) -> None:
        self.block_buffer = [initial_count] * 10
        self.stats = EpisodeTelemetry()

    def update_absence_counter(self, perception: FramePerception) -> None:
        if perception.ball is None:
            self.ball_absence_counter += 1
        else:
            self.ball_absence_counter = 0

    def record_step(self, perception: FramePerception, reward: float, level_saved: bool) -> None:
        self.stats.steps_survived += 1
        self.stats.accumulated_reward += reward
        self._detect_action_jitter()
        
        if level_saved:
            self._tally_broken_blocks(perception.block_count)
            self._detect_paddle_hit(perception)

    def finalize_episode(self, epsilon: float) -> EpisodeTelemetry:
        self.history.episodes.append(len(self.history.episodes) + 1)
        self.history.rewards.append(self.stats.accumulated_reward)
        self.history.survival_frames.append(self.stats.steps_survived)
        self.history.blocks_destroyed.append(self.stats.blocks_destroyed)
        self.history.paddle_hits.append(self.stats.paddle_hits)
        self.history.epsilons.append(epsilon)
        self.history.jitters.append(self.stats.action_jitters)
        return self.stats
        
    def shift_historical_state(self, perception: FramePerception, next_action: int) -> None:
        self.prev_perception = perception
        self.prev_prev_action = self.prev_action
        self.prev_action = next_action

    def _detect_action_jitter(self) -> None:
        jitter_pattern_a = (self.prev_action == 1 and self.prev_prev_action == 2)
        jitter_pattern_b = (self.prev_action == 2 and self.prev_prev_action == 1)
        if jitter_pattern_a or jitter_pattern_b:
            self.stats.action_jitters += 1

    def _tally_broken_blocks(self, current_blocks: int) -> None:
        self.block_buffer.append(current_blocks)
        if len(self.block_buffer) > 10:
            self.block_buffer.pop(0)
            
        stable_count = max(self.block_buffer)
        blocks_broken = self._get_old_blocks() - stable_count
        
        if 0 < blocks_broken <= 3:
            self.stats.blocks_destroyed += blocks_broken

    def _get_old_blocks(self) -> int:
        if self.prev_perception is None:
            return 0
        return self.prev_perception.block_count

    def _detect_paddle_hit(self, perception: FramePerception) -> None:
        if self.prev_perception is None or self.prev_perception.velocity is None:
            return
        if perception.velocity is None or perception.ball is None:
            return
            
        was_falling = self.prev_perception.velocity.y_pos > 0
        is_rising = perception.velocity.y_pos < 0
        is_near_bottom = perception.ball.y_pos > 200  
        
        if was_falling and is_rising and is_near_bottom:
            self.stats.paddle_hits += 1


class ArkanoidOrchestrator:
    """
    Manages the lifecycle, timing, and integration of the Arkanoid agent.
    Delegates all heavy lifting to the injected domain modules.
    """

    def __init__(
        self,
        mode: ExecutionMode,
        emulator: NesEnvironment,
        vision: VisionPipeline,
        brain: ArkanoidBrain,
        dashboard: DashboardManager,
    ) -> None:
        self.mode = mode
        self.emulator = emulator
        self.vision = vision
        self.brain = brain
        self.dashboard = dashboard
        
        self.frame_skip = 4
        self.is_running = True
        self.tracker = TelemetryTracker()
        self.memory_snapshot: bytes = b""

    def execute_loop(self) -> None:
        cli_logger.info(f"Starting Arkanoid lifecycle in {self.mode.value} mode.")
        frame_idx = 0
        try:
            while self.is_running:
                self._process_single_frame(frame_idx)
                frame_idx += 1
        except KeyboardInterrupt:
            cli_logger.info("Keyboard interrupt (Ctrl+C) detected. Exiting...")
        finally:
            self.emulator.hard_reset()

    def _process_single_frame(self, frame_idx: int) -> None:
        level_saved = bool(self.memory_snapshot)
        self.emulator.apply_input(self.tracker.prev_action, self.frame_skip, frame_idx, level_saved)
        frame_matrix = self.emulator.extract_frame()
        
        perception = self.vision.process_game_frame(frame_matrix)
        self._ensure_level_checkpoint(perception)
        
        reward = self._compute_instant_reward(perception)
        if reward <= -100.0:
            self._handle_agent_death(reward)
            return

        self.tracker.record_step(perception, reward, level_saved)
        next_action = self._determine_next_action(perception, reward)
        
        self._render_output_if_needed(frame_idx, frame_matrix, perception, reward)
        self.tracker.shift_historical_state(perception, next_action)

    def _ensure_level_checkpoint(self, perception: FramePerception) -> None:
        if self.memory_snapshot:
            return
            
        objects_present = perception.ball is not None and perception.paddle is not None
        if objects_present:
            self.memory_snapshot = self.emulator.capture_memory_state()
            self.tracker.reset_debounce_buffer(perception.block_count)
            cli_logger.info("Level active. Captured emulator memory state.")

    def _compute_instant_reward(self, perception: FramePerception) -> float:
        if not self.memory_snapshot:
            return 0.0
            
        self.tracker.update_absence_counter(perception)
        return self.brain.calculate_reward(
            perception, 
            self.tracker.ball_absence_counter,
            self.tracker.prev_action,
            self.tracker.prev_prev_action
        )

    def _handle_agent_death(self, terminal_penalty: float) -> None:
        if self.tracker.prev_perception is not None and self.mode != ExecutionMode.SHOWCASE:
            old_state = self.brain.discretizer.discretize(self.tracker.prev_perception)
            self.brain.apply_terminal_penalty(old_state, self.tracker.prev_action, terminal_penalty)
            
        self._log_episode_completion()
        self._revert_environment_state()

    def _determine_next_action(self, perception: FramePerception, reward: float) -> int:
        if perception.paddle is None:
            return 0
            
        curr_state = self.brain.discretizer.discretize(perception)
            
        if self.mode == ExecutionMode.SHOWCASE:
            return self.brain.decide_optimal_action(curr_state)
            
        old_state = None
        if self.tracker.prev_perception is not None:
            old_state = self.brain.discretizer.discretize(self.tracker.prev_perception)
            
        return self.brain.decide_exploratory_action(
            curr_state, old_state, self.tracker.prev_action, reward
        )

    def _render_output_if_needed(
        self, frame_idx: int, frame: np.ndarray, perception: FramePerception, reward: float
    ) -> None:
        if self.mode == ExecutionMode.TRAIN_HEADLESS:
            return
            
        if frame_idx % 5 == 0 or reward != 0:
            curr_state = self.brain.discretizer.discretize(perception)
            q_vals = self.brain.policy.q_table[curr_state].tolist()
            
            keep_running = self.dashboard.tick_realtime(frame, perception, q_vals)
            if not keep_running:
                self.is_running = False
                cli_logger.info("UI closed. Terminating loop gracefully.")

    def _log_episode_completion(self) -> None:
        epsilon = self.brain.decay_exploration_rate(
            self.tracker.stats.steps_survived,
            self.tracker.stats.paddle_hits
        )
        stats = self.tracker.finalize_episode(epsilon)
        self.dashboard.tick_episode(self.tracker.history)
        
        log_payload = {
            "survived": stats.steps_survived,
            "hits": stats.paddle_hits,
            "blocks": stats.blocks_destroyed,
            "reward": stats.accumulated_reward
        }
        
        debug_logger.debug(json.dumps(log_payload))
        if self.mode == ExecutionMode.TRAIN_HEADLESS:
            cli_logger.info(
                f"[EPISODE END] Survived: {stats.steps_survived:04d} | "
                f"Hits: {stats.paddle_hits:02d} | Eps: {epsilon:.4f} | "
                f"Rwd: {stats.accumulated_reward:06.2f}"
            )

    def _revert_environment_state(self) -> None:
        if self.memory_snapshot:
            self.emulator.restore_memory_state(self.memory_snapshot)
        else:
            self.emulator.hard_reset()
            
        self.tracker.reset_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arkanoid RL Agent Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        default="showcase",
        choices=[m.value for m in ExecutionMode],
        help="Selects the execution and rendering trajectory."
    )
    
    args = parser.parse_args()
    selected_mode = ExecutionMode(args.mode)

    if selected_mode == ExecutionMode.SHOWCASE and not os.path.exists("arkanoid_brain.pkl"):
        cli_logger.warning(
            "WARNING: Running in SHOWCASE mode without a trained model! "
            "The agent will not explore and will default to holding LEFT."
        )

    try:
        physics_env = PhysicsEnvironment(left_wall=16.0, right_wall=240.0, paddle_y=212.0)
        vision_config = VisionConfig(ball_threshold=204, paddle_threshold=127, physics=physics_env)
        
        rl_config = RlConfig()
        brain_archive = BrainArchive()
        
        director = ArkanoidOrchestrator(
            mode=selected_mode,
            emulator=NesEnvironment("roms/arkanoid.nes"),
            vision=VisionPipeline(vision_config),
            brain=ArkanoidBrain(config=rl_config, archive=brain_archive),
            dashboard=DashboardManager(headless=(selected_mode == ExecutionMode.TRAIN_HEADLESS))
        )
        
        # Now safely runs until closed or interrupted!
        director.execute_loop() 
        
    except Exception as exc:
        if not hasattr(exc, 'args') or len(exc.args) == 0:
            raise
        offending_val = exc.args[0] if exc.args else "Unknown"
        cli_logger.error(
            f"Fatal error during execution. Offending value: {offending_val}. "
            "Expected valid state progression."
        )
        sys.exit(1)