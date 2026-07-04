import math
import numpy as np
import cv2
from typing import Optional, List, Tuple
from dataclasses import dataclass

from domain.models import Coordinate, FramePerception


@dataclass(frozen=True)
class PhysicsEnvironment:
    left_wall: float
    right_wall: float
    paddle_y: float

@dataclass(frozen=True)
class VisionConfig:
    ball_threshold: int
    paddle_threshold: int
    physics: PhysicsEnvironment
    
@dataclass(frozen=True)
class BoundingRegion:
    area: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float


class VisionPipeline:
    """
    Analyzes raw NES emulator frames to extract physical state data.
    Maintains temporal state to calculate velocities and resolve ambiguities.
    """
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._prev_paddle: Optional[Coordinate] = None
        self._prev_ball: Optional[Coordinate] = None

    def process_game_frame(self, frame_img: np.ndarray) -> FramePerception:
        """
        Coordinates the vision processing pipeline for a single frame.
        Example: pipeline.process_game_frame(emulator.get_image()) -> FramePerception(...)
        """
        self._validate_frame_input(frame_img)
        gray_img = self._ensure_grayscale(frame_img)
        
        ball_pos = self._locate_ball(gray_img)
        paddle_pos = self._locate_paddle(gray_img)
        block_count, density = self._analyze_blocks(gray_img)
        
        velocity, intercept_x = self._calculate_trajectory(ball_pos)
        
        self._step_temporal_state(ball_pos, paddle_pos)
        
        return FramePerception(
            ball=ball_pos,
            paddle=paddle_pos,
            velocity=velocity,
            intercept_x=intercept_x,
            block_count=block_count,
            block_density=density
        )

    def _validate_frame_input(self, frame_img: np.ndarray) -> None:
        if not isinstance(frame_img, np.ndarray):
            raise TypeError(
                f"Invalid frame format: {type(frame_img)}. Expected numpy ndarray."
            )
        if len(frame_img.shape) < 2:
            raise ValueError(
                f"Invalid frame shape: {frame_img.shape}. Expected 2D or 3D image."
            )

    def _ensure_grayscale(self, frame_img: np.ndarray) -> np.ndarray:
        # Downstream detectors should not care whether the frame source
        # is RGB or already grayscale.
        if len(frame_img.shape) == 3 and frame_img.shape[2] == 3:
            return cv2.cvtColor(frame_img, cv2.COLOR_RGB2GRAY)
        return frame_img

    def _segment_bright_pixels(self, gray_img: np.ndarray, threshold: int) -> np.ndarray:
        # The NES Arkanoid left and right walls are grey and can survive thresholding.
        # By forcing the literal edges of the screen to black, we prevent the 
        # paddle from merging with the walls when it moves to the extremes.
        _, binary_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        
        binary_mask[:, :18] = 0
        binary_mask[:, 191:] = 0
        
        return binary_mask

    def _apply_horizontal_closing(self, binary_mask: np.ndarray) -> np.ndarray:
        # Sweeps a horizontal brush over the binary image to connect adjacent blobs.
        # This repairs the broken, disconnected extremities of the Arkanoid paddle.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    def _extract_connected_components(self, binary_mask: np.ndarray) -> List[BoundingRegion]:
        # Wraps OpenCV's connected components to avoid leaking cv2 constants
        # into the geometric evaluation logic.
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            regions.append(BoundingRegion(
                area=stats[i, cv2.CC_STAT_AREA],
                width=stats[i, cv2.CC_STAT_WIDTH],
                height=stats[i, cv2.CC_STAT_HEIGHT],
                centroid_x=centroids[i][0],
                centroid_y=centroids[i][1]
            ))
            
        return regions

    def _locate_ball(self, gray_img: np.ndarray) -> Optional[Coordinate]:
        mask = self._segment_bright_pixels(gray_img, self.config.ball_threshold)
        regions = self._extract_connected_components(mask)
        
        candidates = [r for r in regions if self._is_ball_candidate(r)]
        return self._resolve_primary_target(candidates)

    def _is_ball_candidate(self, region: BoundingRegion) -> bool:
        # The ball is one of the smallest connected components and
        # remains approximately square/circular throughout the game.
        is_right_size = 5 <= region.area <= 15
        
        aspect_ratio = region.width / max(region.height, 1)
        is_circular = 0.6 <= aspect_ratio <= 1.4
        
        return is_right_size and is_circular

    def _resolve_primary_target(self, candidates: List[BoundingRegion]) -> Optional[Coordinate]:
        if not candidates:
            return None
            
        # Multi-Ball Targeting: If there are multiple objects (e.g., 3 balls), 
        # always lock onto the "most dangerous" one (lowest on the screen / Max Y).
        lowest_region = max(candidates, key=lambda r: r.centroid_y)
        return Coordinate(lowest_region.centroid_x, lowest_region.centroid_y)

    def _locate_paddle(self, gray_img: np.ndarray) -> Optional[Coordinate]:
        mask = self._segment_bright_pixels(gray_img, self.config.paddle_threshold)
        closed_mask = self._apply_horizontal_closing(mask)
        regions = self._extract_connected_components(closed_mask)
        
        candidates = [r for r in regions if self._is_paddle_candidate(r)]
        return self._resolve_paddle_ambiguity(candidates)

    def _is_paddle_candidate(self, region: BoundingRegion) -> bool:
        # Location: The paddle is strictly constrained to the bottom area.
        is_at_bottom = 205 < region.centroid_y < 218
        
        # Size: Broad buffer to accommodate the "Expand" (Enlarge) power-up.
        is_right_size = 15 < region.area < 120
        
        # Shape: Horizontal band. Capped at 8.0 to prevent UI lines from triggering.
        aspect_ratio = region.width / max(region.height, 1)
        is_horizontal = 2.0 < aspect_ratio < 8.0
        
        return is_at_bottom and is_right_size and is_horizontal

    def _resolve_paddle_ambiguity(self, candidates: List[BoundingRegion]) -> Optional[Coordinate]:
        if not candidates:
            return None
            
        if self._prev_paddle is None:
            return Coordinate(candidates[0].centroid_x, candidates[0].centroid_y)
            
        # If lasers or explosions trigger a false positive, assume the real paddle 
        # is the one closest to where we last saw it.
        closest = min(
            candidates, 
            key=lambda r: math.hypot(
                r.centroid_x - self._prev_paddle.x_pos, 
                r.centroid_y - self._prev_paddle.y_pos
            )
        )
        return Coordinate(closest.centroid_x, closest.centroid_y)

    def _analyze_blocks(self, gray_img: np.ndarray) -> Tuple[int, int]:
        # Fast vectorized grid sampling to replace nested for-loops.
        # Grid properties: start_x=24, start_y=20, w=16, h=8, cols=11, rows=18
        sample_xs = np.arange(24, 24 + 11 * 16, 16)
        sample_ys = np.arange(20, 20 + 18 * 8, 8)
        
        # Guard against index out of bounds if image is too small
        max_y, max_x = gray_img.shape
        if sample_xs[-1] >= max_x or sample_ys[-1] >= max_y:
            return 0, 1
            
        # Extract pixel intensities at the exact grid intersections
        grid_intensities = gray_img[sample_ys[:, None], sample_xs]
        surviving_blocks = grid_intensities > 50
        
        total_count = int(surviving_blocks.sum())
        
        blocks_left = int(surviving_blocks[:, :6].sum())
        blocks_right = int(surviving_blocks[:, 6:].sum())
        density = 1 if blocks_left >= blocks_right else 2
        
        return total_count, density

    def _calculate_trajectory(self, current_ball: Optional[Coordinate]) -> Tuple[Optional[Coordinate], float]:
        if current_ball is None or self._prev_ball is None:
            return None, -1.0
            
        vx = current_ball.x_pos - self._prev_ball.x_pos
        vy = current_ball.y_pos - self._prev_ball.y_pos
        velocity = Coordinate(vx, vy)
        
        intercept_x = self._predict_intercept(current_ball, vx, vy)
        return velocity, intercept_x

    def _predict_intercept(self, current_ball: Coordinate, vx: float, vy: float) -> float:
        env = self.config.physics
        play_width = env.right_wall - env.left_wall
        
        # SAFETY CLAMP: Ball is moving UP or is too slow.
        # We cannot accurately predict block bounces. Give agent a stable center target.
        if vy <= 0.1 or play_width <= 0:
            return (env.left_wall + env.right_wall) / 2.0
            
        frames_to_impact = (env.paddle_y - current_ball.y_pos) / vy
        raw_x = current_ball.x_pos + (vx * frames_to_impact)
        
        # O(1) Reflection Equation: Folds the raw X back into the screen boundaries
        x_shifted = raw_x - env.left_wall
        crossings = int(math.floor(x_shifted / play_width))
        rem_x = x_shifted % play_width
        
        if crossings % 2 == 0:
            intercept_x = env.left_wall + rem_x
        else:
            intercept_x = env.right_wall - rem_x
            
        # Guarantee intercept never exceeds boundaries due to float rounding
        return max(env.left_wall, min(env.right_wall, intercept_x))

    def _step_temporal_state(
        self, ball_pos: Optional[Coordinate], paddle_pos: Optional[Coordinate]
    ) -> None:
        if paddle_pos is not None:
            self._prev_paddle = paddle_pos
        if ball_pos is not None:
            self._prev_ball = ball_pos