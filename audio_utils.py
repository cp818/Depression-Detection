import pyaudio
import wave
import numpy as np
import logging
from typing import Optional, Tuple, Dict
import os

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Utility class for audio recording and processing."""
    
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1, 
                 chunk_size: int = 1024,
                 format_type: int = pyaudio.paInt16):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            chunk_size: Size of audio chunks
            format_type: Audio format type
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type
        self.audio = None
        self.stream = None
        
    def init_audio(self) -> bool:
        """Initialize PyAudio."""
        try:
            self.audio = pyaudio.PyAudio()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            return False
    
    def start_stream(self, callback=None) -> bool:
        """
        Start audio stream.
        
        Args:
            callback: Optional callback function for audio processing
            
        Returns:
            Boolean indicating success
        """
        if not self.audio:
            if not self.init_audio():
                return False
        
        try:
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                output=False,
                frames_per_buffer=self.chunk_size,
                stream_callback=callback
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            return False
    
    def stop_stream(self) -> None:
        """Stop audio stream and terminate PyAudio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
    
    def convert_to_pcm16(self, audio_data: np.ndarray) -> bytes:
        """
        Convert float audio data to 16-bit PCM.
        
        Args:
            audio_data: Numpy array of float audio data
            
        Returns:
            Bytes of 16-bit PCM audio
        """
        # Ensure the values are between -1 and 1
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        pcm_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to bytes
        return pcm_data.tobytes()
    
    def save_audio(self, frames: list, filename: str = "recorded_audio.wav") -> str:
        """
        Save recorded audio frames to a WAV file.
        
        Args:
            frames: List of audio frames
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if not frames:
            logger.warning("No audio frames to save")
            return ""
        
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format_type))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            logger.info(f"Audio saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return ""
    
    @staticmethod
    def analyze_audio_quality(frames: list) -> Dict[str, float]:
        """
        Analyze audio quality metrics.
        
        Args:
            frames: List of audio frames
            
        Returns:
            Dictionary of audio quality metrics
        """
        if not frames:
            return {"error": "No audio frames to analyze"}
        
        try:
            # Convert frames to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate metrics
            rms = np.sqrt(np.mean(np.square(audio_data)))
            peak = np.max(np.abs(audio_data))
            
            # Signal-to-noise ratio approximation (simplified)
            if rms > 0:
                # Estimate noise as the bottom 10% of samples
                sorted_data = np.sort(np.abs(audio_data))
                noise_level = np.mean(sorted_data[:int(len(sorted_data) * 0.1)])
                snr = 20 * np.log10(rms / max(noise_level, 1e-10))
            else:
                snr = 0
                
            return {
                "rms_level": float(rms),
                "peak_level": float(peak),
                "estimated_snr_db": float(snr)
            }
        except Exception as e:
            logger.error(f"Failed to analyze audio quality: {e}")
            return {"error": str(e)}
