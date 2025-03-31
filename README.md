# Clustering Bioshock Infinite Voice Lines by Speaker using Embeddings

This repo is a personal project to try to cluster audio files by voice. The primary motivation for this project was to create a labeled dataset of voice lines for personal voice training purposes. The audio files are from the game [Bioshock Infinite](https://en.wikipedia.org/wiki/BioShock_Infinite) developed by [Irrational Games](https://en.wikipedia.org/wiki/Irrational_Games) and published by [2K](https://en.wikipedia.org/wiki/2K_(company)) and were extracted from the game by an unkown user in this [Reddit post](https://www.reddit.com/r/Bioshock/comments/70gs87/i_extracted_the_lutece_twins_voicelines_and_all/). The audio files are all unlabeled, meaning there's no direct mapping between each file and the character who spoke it or what is being spoken. Manually sorting these files would be tedious.

To simplify this process I used [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) to generate embeddings for each of the audio files. These embeddings were then clustered using dimensionality reduction techniques like UMAP and t-SNE, with the goal of discovering clusters that correspond to different characters.

### Repository Structure  

- 📂 `all_voicelines_emb/`: ECAPA-TDNN embeddings of original wav files 
- 📂 `all_voicelines_ogg/`: Original audio files extracted from Reddit post  
- 📂 `all_voicelines_wav/`: Original audio files converted into wav format
- 📂 `tmp/`: Temporary files used for clustering
- 📄 `cluster.ipynb`: Clustering and visualization (jupyter-lab recommended)
- 📄 `convert_audio_files.py`: Converts ogg audio files to wav
- 📄 `generate_embedding.py`: Generates ECAPA-TDNN embeddings from wav files
