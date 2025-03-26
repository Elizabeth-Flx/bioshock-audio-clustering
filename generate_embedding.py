import torchaudio



audio_path = "./all_voicelines_wav/"

signal, sr = torchaudio.load(audio_path)

print(signal)
