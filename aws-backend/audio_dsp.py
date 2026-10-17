import io
import math
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly
from pyrnnoise import RNNoise

class AudioDSPPipeline:
    def __init__(self):
        self.rnnoise = RNNoise(sample_rate=48000)
        self.sos_8k = butter(4, [80, 3800], btype="bandpass", fs=8000, output="sos")
        self.sos_16k = butter(4, [80, 7600], btype="bandpass", fs=16000, output="sos")

    def clean_mic_to_16k_pcm_bytes(self, raw_pcm_bytes: bytes, input_sr: int = 16000) -> bytes:
        audio_int16 = np.frombuffer(raw_pcm_bytes, dtype=np.int16)
        if len(audio_int16) == 0:
            return b""

        audio_float = audio_int16.astype(np.float32) / 32768.0

        if input_sr == 8000:
            filtered = sosfilt(self.sos_8k, audio_float)
            audio_48k = resample_poly(filtered, up=6, down=1)
        else:
            filtered = sosfilt(self.sos_16k, audio_float)
            audio_48k = resample_poly(filtered, up=3, down=1)

        audio_48k_int16 = np.clip(audio_48k * 32768.0, -32768, 32767).astype(np.int16).reshape(1, -1)
        denoised_chunks = [frame[0] for _, frame in self.rnnoise.denoise_chunk(audio_48k_int16)]

        if denoised_chunks:
            clean_48k = np.concatenate(denoised_chunks).astype(np.float32) / 32768.0
        else:
            clean_48k = audio_48k

        clean_16k = resample_poly(clean_48k, up=1, down=3).astype(np.float32)
        max_val = np.max(np.abs(clean_16k))
        if max_val > 0.01:
            clean_16k = (clean_16k / max_val) * 0.9

        return np.clip(clean_16k * 32767.0, -32768, 32767).astype(np.int16).tobytes()

    def process_salad_tts_wav(self, wav_bytes: bytes, target_sr: int = 16000) -> bytes:
        samples, orig_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        g = math.gcd(target_sr, orig_sr)
        resampled = resample_poly(samples, up=target_sr // g, down=orig_sr // g)

        if target_sr == 8000:
            resampled = sosfilt(self.sos_8k, resampled)
        else:
            resampled = sosfilt(self.sos_16k, resampled)

        return np.clip(resampled * 32767.0, -32768, 32767).astype(np.int16).tobytes()
