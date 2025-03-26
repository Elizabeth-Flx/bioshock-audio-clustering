import os
from pydub import AudioSegment

audio_path = r"F:\Workspace\bioshock-audio-clustering\all_voicelines_ogg"
output_folder = r"F:\Workspace\bioshock-audio-clustering\all_voicelines_wav2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert all .ogg files to .wav
for file in os.listdir(audio_path):
    ogg_path = os.path.join(audio_path, file)
    wav_path = os.path.join(output_folder, file.split("_English(US)-")[1].replace(".ogg", ".wav"))

    audio = AudioSegment.from_ogg(ogg_path)
    audio.export(wav_path, format="wav")
    print(f"Converted {file} to .wav")

print("Conversion complete!")