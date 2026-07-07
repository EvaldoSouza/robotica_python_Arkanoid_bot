import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

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
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def render_frame(
        self, frame: np.ndarray, perception: FramePerception, q_values: List[float]
    ) -> bool:
        self._validate_inputs(frame, q_values)
        composite = self._construct_composite(frame, perception, q_values)
        cv2.imshow(self.window_name, composite)
        
        return self._poll_ui_events()

    def _construct_composite(
        self, frame: np.ndarray, perception: FramePerception, q_values: List[float]
    ) -> np.ndarray:
        vision_img = self._draw_agent_vision(perception)
        q_chart_img = self._draw_q_values(q_values)
        return np.hstack((frame, vision_img, q_chart_img))

    def _poll_ui_events(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            return False
            
        try:
            is_open = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE) >= 0
            return is_open
        except cv2.error:
            return False

    def _validate_inputs(self, frame: np.ndarray, q_values: List[float]) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Invalid frame format. Offending value: {type(frame)}. "
                "Expected shape: numpy ndarray."
            )
        if len(q_values) != 3:
            raise ValueError(
                f"Invalid Q-values length. Offending value: {len(q_values)}. "
                "Expected shape: List of exactly 3 floats."
            )

    def _draw_agent_vision(self, perception: FramePerception) -> np.ndarray:
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
        
        has_values = not all(q == 0.0 for q in q_values)
        max_q = max(abs(q) for q in q_values) if has_values else 1.0
        max_q = max(max_q, 0.1)
        
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
    """Manages the long-term learning curves using Matplotlib."""
    def __init__(self, session_dir: str = "") -> None:
        self.session_dir = session_dir
        self.window_size = 50
        plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.canvas.manager.set_window_title('RL Training Telemetry')

        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.line_reward_raw, = self.ax1.plot([], [], color='lightgray', label='Raw')
        self.line_reward_avg, = self.ax1.plot([], [], color='blue', linewidth=2, label='Avg')
        self.ax1.set_title('Cumulative Episodic Reward')
        self.ax1.legend()

        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.line_q_raw, = self.ax2.plot([], [], color='lightgray', label='Raw')
        self.line_q_avg, = self.ax2.plot([], [], color='purple', linewidth=2, label='Avg')
        self.ax2.set_ylabel('Avg Max Q-Value', color='purple')
        self.ax2.set_title('Agent Confidence')
        self.ax2.legend()

        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.line_jitters, = self.ax3.plot([], [], color='black', linewidth=2)
        self.ax3.set_title('Agent Jitter Frequency')
        self.ax3.set_ylabel('Jitters / Step')

        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.line_survival_raw, = self.ax4.plot([], [], color='lightgray')
        self.line_survival_avg, = self.ax4.plot([], [], color='cyan', linewidth=2)
        self.ax4.set_title('Survival Duration')
        self.ax4.set_ylabel('Frames Survived')
        
        self.fig.tight_layout()

    def update_metrics(self, history: TelemetryHistory) -> None:
        if not history.episodes:
            return
            
        self._update_title(history)
        self._update_reward(history)
        self._update_learning_metrics(history)
        self._update_smoothness(history)
        self._update_survival(history)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

        # Persist the chart to the designated session folder
        save_path = os.path.join(self.session_dir, "telemetry_chart.png") if self.session_dir else "telemetry_chart.png"
        self.fig.savefig(save_path)

    def _update_title(self, history: TelemetryHistory) -> None:
        latest_eps = history.epsilons[-1] if history.epsilons else 1.0
        title = (
            f"RL Training Telemetry | "
            f"Alpha (LR): {history.alpha} | "
            f"Gamma (Discount): {history.gamma} | "
            f"Epsilon (Explore): {latest_eps:.4f}"
        )
        self.fig.suptitle(title, fontsize=12, fontweight='bold')

    def _smooth_series(self, metric_series: List[float]) -> List[float]:
        if not metric_series:
            return []
            
        smoothed = []
        for i in range(len(metric_series)):
            start_idx = max(0, i - self.window_size + 1)
            window_slice = metric_series[start_idx : i + 1]
            smoothed.append(sum(window_slice) / len(window_slice))
            
        return smoothed

    def _update_reward(self, h: TelemetryHistory) -> None:
        self.line_reward_raw.set_data(h.episodes, h.rewards)
        self.line_reward_avg.set_data(h.episodes, self._smooth_series(h.rewards))
        self.ax1.relim()
        self.ax1.autoscale_view()

    def _update_learning_metrics(self, h: TelemetryHistory) -> None:
        self.line_q_raw.set_data(h.episodes, h.avg_max_q)
        self.line_q_avg.set_data(h.episodes, self._smooth_series(h.avg_max_q))
        self.ax2.relim()
        self.ax2.autoscale_view()

    def _update_smoothness(self, h: TelemetryHistory) -> None:
        self.line_jitters.set_data(h.episodes, self._smooth_series(h.jitters))
        self.ax3.relim()
        self.ax3.autoscale_view()

    def _update_survival(self, h: TelemetryHistory) -> None:
        self.line_survival_raw.set_data(h.episodes, h.survival_frames)
        self.line_survival_avg.set_data(h.episodes, self._smooth_series(h.survival_frames))
        self.ax4.relim()
        self.ax4.autoscale_view()


class TelemetryDashboard:
    """Unified facade to manage both real-time and long-term UI updates."""
    def __init__(self, headless: bool = False, session_dir: str = "") -> None:
        self.headless = headless
        
        # Initialize metrics for both modes so headless still tracks long-term charts
        self.metrics_renderer = MetricsRenderer(session_dir=session_dir)
        
        if not headless:
            self.live_renderer = LiveRenderer()

    def tick_realtime(
        self, frame: np.ndarray, perception: FramePerception, q_values: List[float]
    ) -> bool:
        if self.headless:
            return True
        return self.live_renderer.render_frame(frame, perception, q_values)

    def tick_episode(self, history: TelemetryHistory) -> None:
        # Unconditionally update and save metrics at the end of an episode
        self.metrics_renderer.update_metrics(history)