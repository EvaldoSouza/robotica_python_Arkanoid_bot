import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from dataclasses import dataclass

from domain.models import Coordinate, FramePerception, TelemetryHistory

class LiveRenderer:
    """
    Manages the 60 FPS real-time rendering using optimized OpenCV operations.
    Combines the emulator frame, agent vision, and Q-values into one window.
    """
    def __init__(self, window_name: str = "Arkanoid RL Dashboard") -> None:
        self.window_name = window_name
        self.width = 256
        self.height = 240

    def render_frame(
        self, frame: np.ndarray, perception: FramePerception, q_values: List[float]
    ) -> None:
        self._validate_inputs(frame, q_values)
        
        vision_img = self._draw_agent_vision(perception)
        q_chart_img = self._draw_q_values(q_values)
        
        # Stitch horizontally: [ Game | Vision | Q-Values ]
        composite = np.hstack((frame, vision_img, q_chart_img))
        
        cv2.imshow(self.window_name, composite)
        cv2.waitKey(1)

    def _validate_inputs(self, frame: np.ndarray, q_values: List[float]) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Invalid frame format: {type(frame)}. Expected numpy ndarray."
            )
        if len(q_values) != 3:
            raise ValueError(
                f"Invalid Q-values length: {len(q_values)}. Expected exactly 3."
            )

    def _draw_agent_vision(self, perception: FramePerception) -> np.ndarray:
        # Recreates the false-color vision representation using geometry
        vision = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        cv2.putText(vision, f"Blocks: {perception.block_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
        if perception.paddle is not None:
            self._draw_paddle(vision, perception.paddle)
            
        if perception.ball is not None:
            self._draw_ball_and_trajectory(vision, perception)
            
        return vision

    def _draw_paddle(self, img: np.ndarray, paddle: Coordinate) -> None:
        px, py = int(paddle.x_pos), int(paddle.y_pos)
        cv2.line(img, (px - 15, py), (px + 15, py), (0, 255, 0), 4)

    def _draw_ball_and_trajectory(self, img: np.ndarray, perception: FramePerception) -> None:
        if perception.ball is None:
            return
            
        bx, by = int(perception.ball.x_pos), int(perception.ball.y_pos)
        cv2.circle(img, (bx, by), 3, (255, 0, 0), -1)
        
        intercept = int(perception.intercept_x)
        if intercept > 0:
            cv2.line(img, (bx, by), (intercept, 212), (0, 255, 255), 1)
            cv2.circle(img, (intercept, 212), 4, (0, 0, 255), 2)

    def _draw_q_values(self, q_values: List[float]) -> np.ndarray:
        chart = np.zeros((self.height, 150, 3), dtype=np.uint8)
        labels = ["Left", "Stay", "Right"]
        
        max_q = max(abs(q) for q in q_values) if any(q_values) else 1.0
        max_q = max(max_q, 0.1) # Prevent divide by zero
        
        for i, (q, label) in enumerate(zip(q_values, labels)):
            self._draw_single_bar(chart, i, q, max_q, label)
            
        return chart

    def _draw_single_bar(
        self, img: np.ndarray, idx: int, value: float, max_val: float, label: str
    ) -> None:
        bar_x = 20 + (idx * 40)
        center_y = self.height // 2
        
        normalized_h = int((abs(value) / max_val) * (self.height // 2 - 20))
        color = (0, 255, 0) if value >= 0 else (0, 0, 255)
        
        if value >= 0:
            cv2.rectangle(img, (bar_x, center_y - normalized_h), (bar_x + 20, center_y), color, -1)
        else:
            cv2.rectangle(img, (bar_x, center_y), (bar_x + 20, center_y + normalized_h), color, -1)
            
        cv2.putText(img, label, (bar_x - 5, self.height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


class MetricsRenderer:
    """
    Manages the long-term learning curves using Matplotlib.
    Operates in interactive mode to avoid blocking the main thread.
    """
    def __init__(self) -> None:
        self.window_size = 50
        plt.ion() # Enable non-blocking interactive mode
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.canvas.manager.set_window_title('RL Training Telemetry')

    def update_metrics(self, history: TelemetryHistory) -> None:
        if not history.episodes:
            return
            
        self.fig.clf()
        
        self._plot_reward(history)
        self._plot_task_performance(history)
        self._plot_smoothness(history)
        self._plot_survival(history)
        
        self.fig.tight_layout()
        plt.pause(0.001) # Flush GUI events without stalling

    def _smooth_series(self, data: List[float]) -> List[float]:
        if not data:
            return []
            
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - self.window_size + 1)
            window_slice = data[start_idx : i + 1]
            smoothed.append(sum(window_slice) / len(window_slice))
            
        return smoothed

    def _plot_reward(self, h: TelemetryHistory) -> None:
        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(h.episodes, h.rewards, color='lightgray', label='Raw')
        ax.plot(h.episodes, self._smooth_series(h.rewards), color='blue', linewidth=2, label='Avg')
        ax.set_title('Cumulative Episodic Reward')
        ax.legend()

    def _plot_task_performance(self, h: TelemetryHistory) -> None:
        ax1 = self.fig.add_subplot(2, 2, 2)
        ax1.plot(h.episodes, self._smooth_series(h.blocks_destroyed), color='green', linewidth=2)
        ax1.set_ylabel('Blocks Broken (Avg)', color='green')
        
        ax2 = ax1.twinx()
        ax2.plot(h.episodes, h.epsilons, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel('Exploration Rate (Epsilon)', color='red')
        ax1.set_title('Task Execution vs Exploration')

    def _plot_smoothness(self, h: TelemetryHistory) -> None:
        ax = self.fig.add_subplot(2, 2, 3)
        ax.plot(h.episodes, self._smooth_series(h.jitters), color='black', linewidth=2)
        ax.set_title('Agent Jitter Frequency')
        ax.set_ylabel('Jitters / Step')

    def _plot_survival(self, h: TelemetryHistory) -> None:
        ax = self.fig.add_subplot(2, 2, 4)
        ax.plot(h.episodes, h.survival_frames, color='lightgray')
        ax.plot(h.episodes, self._smooth_series(h.survival_frames), color='cyan', linewidth=2)
        ax.set_title('Survival Duration')
        ax.set_ylabel('Frames Survived')


class DashboardManager:
    """
    Unified facade to manage both real-time and long-term UI updates.
    Replaces init_dashboard, update_dashboard, and plot_learning_curve.
    """
    def __init__(self, headless: bool = False) -> None:
        self.headless = headless
        if not headless:
            self.live_renderer = LiveRenderer()
            self.metrics_renderer = MetricsRenderer()

    def tick_realtime(
        self, frame: np.ndarray, perception: FramePerception, q_values: List[float]
    ) -> None:
        """Called every frame to update the fast CV2 UI."""
        if self.headless:
            return
        self.live_renderer.render_frame(frame, perception, q_values)

    def tick_episode(self, history: TelemetryHistory) -> None:
        """Called only at the end of an episode to update charts."""
        if self.headless:
            return
        self.metrics_renderer.update_metrics(history)