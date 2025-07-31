# LyricNet: Generative Songwriting with LSTM and MIDI Context

This project explores the intersection of natural language generation and music understanding. It uses deep learning to generate song lyrics conditioned on melodic input, combining linguistic patterns from real-world lyrics with musical context extracted from MIDI files.

## 🎯 Objective

To develop and compare two models for generating song lyrics:
- **Model 1** uses a static, whole-track MIDI embedding.
- **Model 2** incorporates time-aligned, segment-wise MIDI features for each lyric token.

## 🧠 Model Architecture

Both models use a dual-layer LSTM architecture:
- Input: 300D Word2Vec embeddings + MIDI features (static 115D or time-aligned 17D).
- Two LSTM layers followed by a linear + dropout layer.
- Output: Vocabulary distribution using softmax.

## 🎼 MIDI Feature Extraction

- **Static (Model 1):** One MIDI vector per track (beat stats, pitch, tempo, drums, etc.)
- **Time-aligned (Model 2):** MIDI segment features per word (note counts, chroma, rhythm, velocity)

## 🧪 Training & Optimization

- Optimized using **Optuna** across 63 configurations.
- Key hyperparameters: Learning rate, LSTM size, dropout, weight decay.
- Loss: Cross-entropy
- Optimizer: Adam
- Early stopping and logging via TensorBoard + W&B

## 📊 Evaluation Metrics

- **Cosine Similarity** (semantic coherence)
- **Jaccard Similarity** (lexical overlap)
- **Levenshtein Similarity** (edit distance)
- **Polarity Alignment** (sentiment preservation via VADER)

## 🪄 Lyric Generation Pipeline

- Seeded with various strategies: random word, most/least common word, original test lyric start.
- Generates up to 200 tokens per song.
- Applies stochastic sampling with multinomial softmax.
- Hidden states are periodically reset to maintain topicality.

## 🔍 Key Findings

- **Model 2** significantly outperforms **Model 1** across all metrics.
- Time-aligned MIDI features result in better sentiment fidelity, higher lexical and semantic overlap.
- Rare, neutral seed words yield the most consistent and high-quality generations.

## 📁 Project Structure

```bash
├── code.py                # Model definition and training scripts
├── report.pdf             # Detailed project report with methodology & results
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── midi/              # MIDI files for each song
