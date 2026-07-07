from enum import IntFlag
from typing import Optional
from dataclasses import dataclass

class NesButton(IntFlag):
    """
    Standard NES controller bitmask mapping for nes-py.
    Provides O(1) bitwise operations instead of array allocations.
    """
    NONE = 0
    A = 1 << 0
    B = 1 << 1
    SELECT = 1 << 2
    START = 1 << 3
    UP = 1 << 4
    DOWN = 1 << 5
    LEFT = 1 << 6
    RIGHT = 1 << 7

@dataclass(frozen=True)
class MotorConfig:
    deadzone: float

DEFAULT_INTERCEPT_X = 105.0
MENU_BYPASS_FRAME_WINDOW = 5

def translate_rl_action(action_idx: int, frame_counter: int, level_saved: bool) -> int:
    """
    Maps discrete action indices to hardware bitmasks.
    Example: translate_rl_action(1, 120, True) -> NesButton.LEFT | NesButton.A
    """
    if action_idx < 0 or action_idx > 3:
        raise ValueError(
            f"Invalid action index. Offending value: {action_idx}. "
            "Expected shape: Integer between 0 and 3."
        )

    if action_idx == 0:
        return _evaluate_menu_bypass(frame_counter, level_saved)
    if action_idx == 1:
        return NesButton.LEFT | NesButton.A
    if action_idx == 2:
        return NesButton.RIGHT | NesButton.A
        
    return NesButton.A

def _evaluate_menu_bypass(frame_counter: int, level_saved: bool) -> int:
    is_start_window = (frame_counter % 60) < MENU_BYPASS_FRAME_WINDOW
    
    if not level_saved and is_start_window:
        return NesButton.START
        
    return NesButton.NONE

def calculate_heuristic_input(
    paddle_x: Optional[float], 
    intercept_x: float, 
    frame_counter: int, 
    config: MotorConfig
) -> int:
    """Steers the paddle directly towards the predicted ball intercept."""
    if paddle_x is None:
        return _evaluate_menu_bypass(frame_counter, False)
        
    target_x = intercept_x if intercept_x != -1.0 else DEFAULT_INTERCEPT_X
    return _steer_towards_target(paddle_x, target_x, config.deadzone)

def _steer_towards_target(paddle_x: float, target_x: float, deadzone: float) -> int:
    if paddle_x < (target_x - deadzone):
        return NesButton.RIGHT | NesButton.A
        
    if paddle_x > (target_x + deadzone):
        return NesButton.LEFT | NesButton.A
        
    return NesButton.A