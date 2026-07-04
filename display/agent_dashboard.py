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
        # Pre-initialize the window so properties are immediately valid to the OS
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def render_frame(
        self, frame: np.ndarray, perception: FramePerception, q_values: List[float]
    ) -> bool:
        self._validate_inputs(frame, q_values)
        
        vision_img = self._draw_agent_vision(perception)
        q_chart_img = self._draw_q_values(q_values)
        
        # Stitch horizontally: [ Game | Vision | Q-Values ]
        composite = np.hstack((frame, vision_img, q_chart_img))
        
        cv2.imshow(self.window_name, composite)
        
        # 1ms delay allows OpenCV to process GUI events (including the 'X' button)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print(f"Key pressed: {key}")
            return False # User pressed ESC or 'q'
            
        # NATIVE X-BUTTON DETECTION
        # WND_PROP_VISIBLE is completely broken on some Linux Qt backends and always returns -1.
        # Instead, we check WND_PROP_AUTOSIZE. Because we initialized the window with WINDOW_AUTOSIZE,
        # it returns 1.0 as long as the window exists, and -1.0 the moment the 'X' button is clicked.
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE) < 0:
                print("Window is closed")
                return False
        except cv2.error as e:
            # If the C++ window handle is fully destroyed, getting a property throws an error
            print(f"Closed with error: {e}")
            return False
            
        return True

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

        # Initialize axes and lines once to prevent memory leaks
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.line_reward_raw, = self.ax1.plot([], [], color='lightgray', label='Raw')
        self.line_reward_avg, = self.ax1.plot([], [], color='blue', linewidth=2, label='Avg')
        self.ax1.set_title('Cumulative Episodic Reward')
        self.ax1.legend()

        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.line_q_raw, = self.ax2.plot([], [], color='lightgray', label='Raw')
        self.line_q_avg, = self.ax2.plot([], [], color='purple', linewidth=2, label='Avg')
        self.ax2.set_ylabel('Avg Max Q-Value', color='purple')
        self.ax2.set_title('Agent Confidence (Expected Future Reward)')
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
            
        self._update_reward(history)
        self._update_learning_metrics(history)
        self._update_smoothness(history)
        self._update_survival(history)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    def _smooth_series(self, data: List[float]) -> List[float]:
        if not data:
            return []
            
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - self.window_size + 1)
            window_slice = data[start_idx : i + 1]
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
    ) -> bool:
        """Called every frame to update the fast CV2 UI."""
        if self.headless:
            return True
        return self.live_renderer.render_frame(frame, perception, q_values)

    def tick_episode(self, history: TelemetryHistory) -> None:
        """Called only at the end of an episode to update charts."""
        if self.headless:
            return
        self.metrics_renderer.update_metrics(history)