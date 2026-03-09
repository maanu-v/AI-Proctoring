"""
Speaker Detection using Resemblyzer.
Compares speaker voice embeddings against a reference voiceprint
to detect if someone other than the enrolled student is speaking.

Supports two enrollment modes:
  - explicit: Voice sample provided at session start
  - auto: First N seconds of detected speech used as reference (auto-calibration)
"""

import numpy as np
import logging
from resemblyzer import VoiceEncoder, preprocess_wav

logger = logging.getLogger(__name__)


class SpeakerDetector:
    """
    Extracts d-vector speaker embeddings and compares against a reference.

    Audio requirements (same as VAD):
        - Format: Raw PCM, 16-bit signed integers (little-endian)
        - Sample rate: 16000 Hz
        - Channels: Mono (1 channel)
    """

    def __init__(self, similarity_threshold: float = 0.75, sample_rate: int = 16000,
                 enrollment_mode: str = "auto", calibration_duration: float = 3.0):
        """
        Args:
            similarity_threshold: Cosine similarity below which a speaker is flagged as unknown.
            sample_rate: Audio sample rate in Hz.
            enrollment_mode: "auto" (calibrate from first speech) or "explicit" (provide sample).
            calibration_duration: Seconds of speech to collect before auto-calibration completes.
        """
        self.encoder = VoiceEncoder()
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate
        self.enrollment_mode = enrollment_mode
        self.calibration_duration = calibration_duration

        # Reference embedding (set via enroll() or auto-calibration)
        self.reference_embedding = None
        self.is_calibrated = False

        # Auto-calibration buffer
        self._calibration_buffer = []
        self._calibration_samples = 0

    def enroll(self, audio_bytes: bytes) -> bool:
        """
        Explicitly enroll a speaker's voiceprint from a voice sample.

        Args:
            audio_bytes: Raw PCM audio of the student speaking.

        Returns:
            True if enrollment succeeded, False otherwise.
        """
        embedding = self._extract_embedding(audio_bytes)
        if embedding is not None:
            self.reference_embedding = embedding
            self.is_calibrated = True
            logger.info("Speaker enrolled successfully (explicit mode)")
            return True
        logger.warning("Speaker enrollment failed — could not extract embedding")
        return False

    def process(self, audio_bytes: bytes) -> dict:
        """
        Process audio and compare speaker against reference.

        If not calibrated and in auto mode, audio is buffered for calibration.

        Args:
            audio_bytes: Raw PCM audio (16-bit, 16kHz, mono) — should contain speech.

        Returns:
            dict with:
                - is_calibrated: bool
                - calibration_progress: float (0.0 to 1.0, only during auto-calibration)
                - similarity: float (cosine similarity to reference, -1 if not calibrated)
                - is_reference_speaker: bool
                - unknown_speaker_detected: bool
        """
        # Auto-calibration: buffer speech until we have enough
        if not self.is_calibrated and self.enrollment_mode == "auto":
            return self._handle_auto_calibration(audio_bytes)

        # Not calibrated and explicit mode — can't compare
        if not self.is_calibrated:
            return {
                "is_calibrated": False,
                "calibration_progress": 0.0,
                "similarity": -1.0,
                "is_reference_speaker": False,
                "unknown_speaker_detected": False,
            }

        # Extract embedding from current audio
        current_embedding = self._extract_embedding(audio_bytes)
        if current_embedding is None:
            return {
                "is_calibrated": True,
                "calibration_progress": 1.0,
                "similarity": -1.0,
                "is_reference_speaker": False,
                "unknown_speaker_detected": False,
            }

        # Compare
        similarity = float(np.dot(self.reference_embedding, current_embedding) /
                          (np.linalg.norm(self.reference_embedding) * np.linalg.norm(current_embedding)))

        is_reference = similarity >= self.similarity_threshold

        return {
            "is_calibrated": True,
            "calibration_progress": 1.0,
            "similarity": round(similarity, 4),
            "is_reference_speaker": is_reference,
            "unknown_speaker_detected": not is_reference,
        }

    def _handle_auto_calibration(self, audio_bytes: bytes) -> dict:
        """Buffer speech audio and calibrate when we have enough."""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._calibration_buffer.append(samples)
        self._calibration_samples += len(samples)

        total_seconds = self._calibration_samples / self.sample_rate
        progress = min(total_seconds / self.calibration_duration, 1.0)

        if total_seconds >= self.calibration_duration:
            # Enough audio collected — extract reference embedding
            all_audio = np.concatenate(self._calibration_buffer)
            wav = preprocess_wav(all_audio, source_sr=self.sample_rate)
            self.reference_embedding = self.encoder.embed_utterance(wav)
            self.is_calibrated = True
            self._calibration_buffer = []
            self._calibration_samples = 0
            logger.info("Speaker auto-calibration complete — voiceprint enrolled")

        return {
            "is_calibrated": self.is_calibrated,
            "calibration_progress": round(progress, 2),
            "similarity": -1.0,
            "is_reference_speaker": False,
            "unknown_speaker_detected": False,
        }

    def _extract_embedding(self, audio_bytes: bytes) -> np.ndarray | None:
        """
        Extract a d-vector embedding from raw PCM audio.

        Returns:
            256-dim numpy array or None if audio is too short.
        """
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            wav = preprocess_wav(samples, source_sr=self.sample_rate)
            if len(wav) < self.sample_rate * 0.5:  # need at least 0.5s of audio
                return None
            return self.encoder.embed_utterance(wav)
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return None

    def reset(self):
        """Reset all state (keeps enrollment_mode and thresholds)."""
        self.reference_embedding = None
        self.is_calibrated = False
        self._calibration_buffer = []
        self._calibration_samples = 0
