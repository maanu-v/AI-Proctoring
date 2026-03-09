"""
Voice Activity Detection (VAD) using webrtcvad.
Detects speech presence in raw PCM audio chunks.
"""

import math
import struct
import time
import webrtcvad


class VADProcessor:
    """
    Processes raw PCM audio and detects speech using Google's WebRTC VAD.
    
    Audio requirements:
        - Format: Raw PCM, 16-bit signed integers (little-endian)
        - Sample rate: 16000 Hz
        - Channels: Mono (1 channel)
        - Frame duration: 20ms (640 bytes per frame)
    """

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000, frame_duration_ms: int = 20,
                 speech_buffer_duration: float = 2.0, energy_threshold: float = 0.02):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        # bytes per frame: sample_rate * frame_duration_ms / 1000 * 2 (16-bit = 2 bytes)
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2

        # RMS energy threshold (normalized 0-1) to filter ambient noise like fans/AC
        # Speech typically has RMS > 0.02, fan noise is usually < 0.01
        self.energy_threshold = energy_threshold

        # State tracking
        self.speech_start_time = None
        self.speech_duration = 0.0
        self.total_speech_segments = 0
        self._was_speaking = False

        # Speech buffer for speaker detection
        self._speech_buffer_duration = speech_buffer_duration
        self._speech_buffer = bytearray()
        self._speech_buffer_bytes_needed = int(sample_rate * speech_buffer_duration) * 2  # 16-bit

    def process(self, audio_bytes: bytes) -> dict:
        """
        Process a chunk of raw PCM audio bytes.

        Args:
            audio_bytes: Raw PCM audio (16-bit, 16kHz, mono)

        Returns:
            dict with keys:
                - is_speech: bool - whether speech was detected in this chunk
                - speech_duration: float - continuous speech duration in seconds
                - speech_segments: int - total speech segments detected so far
                - frames_with_speech: int - number of 20ms frames with speech
                - total_frames: int - total 20ms frames processed in this chunk
                - speech_buffer_ready: bool - whether the speech buffer has enough audio for speaker detection
        """
        # Pre-filter: check RMS energy of the entire chunk
        chunk_rms = self._compute_rms(audio_bytes)
        if chunk_rms < self.energy_threshold:
            # Audio energy too low — this is ambient noise, skip VAD entirely
            is_speech = False
            speech_frame_count = 0
            total_frames = max(1, len(audio_bytes) // self.frame_size)
        else:
            speech_frame_count = 0
            total_frames = 0

            for i in range(0, len(audio_bytes) - self.frame_size + 1, self.frame_size):
                frame = audio_bytes[i:i + self.frame_size]
                total_frames += 1
                if self.vad.is_speech(frame, self.sample_rate):
                    speech_frame_count += 1

            # Determine if this chunk is speech (majority vote)
            is_speech = speech_frame_count > (total_frames / 2) if total_frames > 0 else False

        # Track continuous speech duration
        if is_speech:
            if self.speech_start_time is None:
                self.speech_start_time = time.time()
                if not self._was_speaking:
                    self.total_speech_segments += 1
            self.speech_duration = time.time() - self.speech_start_time
            # Accumulate speech audio into buffer
            self._speech_buffer.extend(audio_bytes)
        else:
            self.speech_start_time = None
            self.speech_duration = 0.0
            # Clear buffer when speech stops
            self._speech_buffer = bytearray()

        self._was_speaking = is_speech

        return {
            "is_speech": is_speech,
            "speech_duration": round(self.speech_duration, 2),
            "speech_segments": self.total_speech_segments,
            "frames_with_speech": speech_frame_count,
            "total_frames": total_frames,
            "speech_buffer_ready": len(self._speech_buffer) >= self._speech_buffer_bytes_needed,
        }

    def get_speech_buffer(self) -> bytes:
        """Get the accumulated speech buffer and clear it."""
        buf = bytes(self._speech_buffer)
        self._speech_buffer = bytearray()
        return buf

    @staticmethod
    def _compute_rms(audio_bytes: bytes) -> float:
        """Compute normalized RMS energy of 16-bit PCM audio. Returns 0.0-1.0."""
        n_samples = len(audio_bytes) // 2
        if n_samples == 0:
            return 0.0
        samples = struct.unpack(f'<{n_samples}h', audio_bytes[:n_samples * 2])
        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / n_samples)
        return rms / 32768.0  # normalize to 0-1

    def reset(self):
        """Reset all tracking state."""
        self.speech_start_time = None
        self.speech_duration = 0.0
        self.total_speech_segments = 0
        self._was_speaking = False
        self._speech_buffer = bytearray()
