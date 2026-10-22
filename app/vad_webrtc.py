from __future__ import annotations

from dataclasses import dataclass
import webrtcvad


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20            # 10/20/30 ms supported by WebRTC VAD
    aggressiveness: int = 2       # 0..3 (3 most aggressive)
    min_voice_ms: int = 120       # voice needed to consider "speaking"
    hangover_ms: int = 700        # silence needed to consider "stopped"


class WebRTCVAD:
    """
    WebRTC VAD (py-webrtcvad).
    Feed PCM16LE mono at 16000 Hz.
    Call update(pcm16_bytes) repeatedly.
    Use is_speaking + silence timer for auto-stop or barge-in.
    """
    def __init__(self, cfg: VADConfig = VADConfig()):
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.aggressiveness)
        self.bytes_per_frame = int(cfg.sample_rate * (cfg.frame_ms / 1000.0) * 2)  # int16 bytes
        self._buf = bytearray()

        self._voice_ms_acc = 0
        self._silence_ms_acc = 0
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def reset(self):
        self._buf = bytearray()
        self._voice_ms_acc = 0
        self._silence_ms_acc = 0
        self._speaking = False

    def update(self, pcm16: bytes) -> tuple[bool, bool]:
        """
        Returns: (speech_detected_in_this_chunk, speaking_state_changed)
        """
        if not pcm16:
            return False, False

        self._buf.extend(pcm16)
        speech_any = False
        state_changed = False

        while len(self._buf) >= self.bytes_per_frame:
            frame = bytes(self._buf[: self.bytes_per_frame])
            del self._buf[: self.bytes_per_frame]

            is_speech = self.vad.is_speech(frame, self.cfg.sample_rate)
            speech_any = speech_any or is_speech

            if is_speech:
                self._voice_ms_acc += self.cfg.frame_ms
                self._silence_ms_acc = 0
            else:
                self._silence_ms_acc += self.cfg.frame_ms
                self._voice_ms_acc = 0

            if (not self._speaking) and self._voice_ms_acc >= self.cfg.min_voice_ms:
                self._speaking = True
                state_changed = True

            if self._speaking and self._silence_ms_acc >= self.cfg.hangover_ms:
                self._speaking = False
                state_changed = True

        return speech_any, state_changed
