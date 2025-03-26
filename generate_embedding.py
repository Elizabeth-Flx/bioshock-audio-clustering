import torchaudio
import os
# os.environ["SB_DISABLE_SYMLINKS"] = "1"  # Completely disable symlinks
# os.environ["SB_DOWNLOAD_STRATEGY"] = "COPY"

from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

wav_path = r"F:\Workspace\bioshock-audio-clustering\all_voicelines_wav"
emb_path = r"F:\Workspace\bioshock-audio-clustering\all_voicelines_emb"

# Create output folder if it doesn't exist
os.makedirs(emb_path, exist_ok=True)


# Load the model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir_spkrec-ecapa-voxceleb",
    # use_auth_token=False,
    # local_files_only=False,
    # local_strategy=LocalStrategy.COPY  # Use COPY instead of SYMLINK)
)

# Embed wav file
def generate_embedding(wav_file):
    signal, sr = torchaudio.load(wav_file)
    embedding = classifier.encode_batch(signal)
    return embedding

 
wav_files = [os.path.join(wav_path, file) for file in os.listdir(wav_path)]
print(wav_files[0]) 


print(generate_embedding(wav_files[0]))

