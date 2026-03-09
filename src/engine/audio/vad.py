"""
Voice Activity Detection (VAD) using webrtcvad.
Detects speech presence in raw PCM audio chunks.
"""

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

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000, frame_duration_ms: int = 20):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        # bytes per frame: sample_rate * frame_duration_ms / 1000 * 2 (16-bit = 2 bytes)
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2

        # State tracking
        self.speech_start_time = None
        self.speech_duration = 0.0
        self.total_speech_segments = 0
        self._was_speaking = False

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
        """
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
        else:
            self.speech_start_time = None
            self.speech_duration = 0.0

        self._was_speaking = is_speech

        return {
            "is_speech": is_speech,
            "speech_duration": round(self.speech_duration, 2),
            "speech_segments": self.total_speech_segments,
            "frames_with_speech": speech_frame_count,
            "total_frames": total_frames,
        }

    def reset(self):
        """Reset all tracking state."""
        self.speech_start_time = None
        self.speech_duration = 0.0
        self.total_speech_segments = 0
        self._was_speaking = False
