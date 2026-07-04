import numpy as np
from nes_py import NESEnv
from emulator.emulator_translator import translate_rl_action

class NesEnvironment:
    """
    Adapter for the NES emulator to match the orchestrator's expectations.
    Isolates third-party nes-py dependencies from the core domain logic.
    """
    
    def __init__(self, rom_path: str) -> None:
        self.rom_path = rom_path
        self._env = NESEnv(self.rom_path)
        self._last_frame = self._env.reset()
        
    def apply_input(
        self, action_idx: int, frame_skip: int, frame_counter: int, level_saved: bool
    ) -> None:
        bitmask = translate_rl_action(action_idx, frame_counter, level_saved)
        
        for _ in range(frame_skip):
            frame, _, done, _ = self._env.step(bitmask)
            self._last_frame = frame
            if done:
                self._last_frame = self._env.reset()
                
    def extract_frame(self) -> np.ndarray:
        if self._last_frame is None:
            raise RuntimeError(
                f"No frame available. Expected numpy array, got {type(self._last_frame)}."
            )
        return self._last_frame
        
    def capture_memory_state(self) -> bytes:
        try:
            # nes-py wraps the C++ emulator which exposes these protected methods
            if hasattr(self._env.unwrapped, '_backup'):
                self._env.unwrapped._backup()
                return b"saved_internally"
            if hasattr(self._env, 'get_state'):
                return self._env.get_state()
        except Exception:
            pass
        return b""
        
    def restore_memory_state(self, state: bytes) -> None:
        if not state:
            return
            
        try:
            if hasattr(self._env.unwrapped, '_restore'):
                self._env.unwrapped._restore()
            elif hasattr(self._env, 'set_state'):
                self._env.set_state(state)
        except Exception:
            self.hard_reset()
            
    def hard_reset(self) -> None:
        self._last_frame = self._env.reset()