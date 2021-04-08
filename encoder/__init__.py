name = "encoder"

from encoder.audio import preprocess_wav, wav_to_mel_spectrogram, trim_long_silences, \
    normalize_volume
from encoder.hparams import sampling_rate
from encoder.voice_encoder import VoiceEncoder
