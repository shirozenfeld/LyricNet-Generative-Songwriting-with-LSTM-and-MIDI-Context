# Imports
import pandas as pd
import numpy as np
import time
import re
import os
import pretty_midi
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gensim.downloader
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
import itertools
import wandb
import torch
from torch.utils.data.dataloader import default_collate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load Word2Vec Model
word2vec = gensim.downloader.load('word2vec-google-news-300')


# # Load Data

train_set = pd.read_csv('./lyrics_train_set.csv', header=None, usecols=[0, 1, 2])
train_set.columns = ['Artist', 'Song_name', 'Lyrics']
test_set = pd.read_csv('./lyrics_test_set.csv', header=None, usecols=[0, 1, 2])
test_set.columns = ['Artist', 'Song_name', 'Lyrics']


# # Data Analysis
def Lexical_Richness(data):
  data['Unique_Words'] = data['Lyrics'].apply(lambda x: len(set(x.split())))
  data['Total_Words'] = data['Lyrics'].apply(lambda x: len(x.split()))
  data['Lexical_Richness'] = data['Unique_Words'] / data['Total_Words']

  top_lexical_richness = data[['Artist', 'Song_name', 'Lexical_Richness']].sort_values(by='Lexical_Richness', ascending=False).head(10)

  # Plotting
  plt.figure(figsize=(10, 8))
  plt.barh(top_lexical_richness['Song_name'], top_lexical_richness['Lexical_Richness'], color='skyblue')
  plt.xlabel('Lexical Richness')
  plt.ylabel('Song Name')
  plt.title('Top 10 Songs by Lexical Richness')
  plt.gca().invert_yaxis()  
  plt.show()

def count_words(lyrics):
    return len(lyrics.split())

train_set['Word_Count'] = train_set['Lyrics'].apply(count_words)
test_set['Word_Count'] = test_set['Lyrics'].apply(count_words)

mean_word_count_train = train_set['Word_Count'].mean()
mean_word_count_test = test_set['Word_Count'].mean()

print(f"Average number of words per song in the Train Set: {mean_word_count_train:.2f}")
print(f"Average number of words per song in the Test Set: {mean_word_count_test:.2f}")

num_artists_train = train_set['Artist'].nunique()
num_artists_test = test_set['Artist'].nunique()

print(f"Number of unique artists in the Train Set: {num_artists_train}")
print(f"Number of unique artists in the Test Set: {num_artists_test}")

# Updated ngram_analysis and topic_modeling with extra length filter

from collections import Counter
import re
from gensim import corpora, models
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# stopwords set
stops = set(ENGLISH_STOP_WORDS)

# simple regex tokenizer
def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def ngram_analysis(data, n=2, top_k=20):
    """
    Extracts top_k n-grams filtering out stopwords AND tokens shorter than 3 characters.
    """
    counter = Counter()
    for txt in data['Lyrics']:
        tokens = [t for t in simple_tokenize(txt) if t not in stops and len(t) > 2]
        for gram in zip(*[tokens[i:] for i in range(n)]):
            counter[gram] += 1
    return counter.most_common(top_k)

def topic_modeling(data, num_topics=5):
    """
    Builds an LDA model on filtered tokens (stopwords removed and tokens >= 3 chars).
    """
    texts = [
        [t for t in simple_tokenize(txt) if t not in stops and len(t) > 2]
        for txt in data['Lyrics']
    ]
    dictionary = corpora.Dictionary(texts)
    corpus     = [dictionary.doc2bow(text) for text in texts]
    lda        = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda.print_topics()

# Usage example
import pandas as pd
data = pd.read_csv('lyrics_train_set.csv')
print("Filtered Top Bigrams:", ngram_analysis(data))
print("LDA Topics (filtered):", topic_modeling(data))



# # Create Vocabularies
# 


vocab = set()
filters = '!"#$%()*+&-/:;<=>?@[\\]^_`{|}~\t\n'
for lyrics in train_set.Lyrics.tolist():
    words = re.sub(f'[{filters}]', '', lyrics).split()
    vocab.update(words)
vocab.discard('')
vocab_size = len(vocab) + 1

word2index = {w: i for i, w in enumerate(vocab)}
index2word = {i: w for w, i in word2index.items()}
print(vocab_size)


# Convert lyrics to indices
# 


train_set['tokens'] = train_set['Lyrics'].apply(lambda x: [word2index.get(word, 0) for word in re.sub(f'[{filters}]', '', x).split()])
test_set['tokens'] = test_set['Lyrics'].apply(lambda x: [word2index.get(word, 0) for word in re.sub(f'[{filters}]', '', x).split()])


# # Song Data Class

class SongDataset(Dataset):
    """
    A dataset class for handling song data, which includes lyrics and MIDI music features.
    
    Attributes:
        data (DataFrame): A pandas DataFrame containing song metadata and lyrics.
        midi_path (str): Directory path where MIDI files are stored.
        word2vec (dict): Pre-trained word2vec model for converting words to vectors.
        vocab (list or dict): Vocabulary used in the dataset.
        method (str): Specifies the method to extract features from MIDI files.
        word2index (dict): Dictionary mapping words to their indices.
        index2word (dict): Dictionary mapping indices back to words.
    """
    
    def __init__(self, df, word2vec, word2index, vocab, midi_path, method):
        """
        df: DataFrame with ['Artist','Song_name','Lyrics']
        method: 'model1' or 'model2'
        midi_path: root folder for MIDI files
        """
        self.data      = df.reset_index(drop=True)
        self.word2vec  = word2vec
        self.vocab     = vocab
        self.midi_path = midi_path
        self.method    = method
        self.word2index = word2index
        self.index2word = {i: w for w, i in word2index.items()}


        # ─── CACHE MIDI FEATURES ────────────────────────────────────────────
        # model1_cache: {(artist,song): Tensor[115]}
        # model2_cache: {idx: Tensor[T_i,17]}
        self.model1_cache = {}
        self.model2_cache = {}
        self.model2_cache_dir = os.path.join(self.midi_path, "model2_cache")
        os.makedirs(self.model2_cache_dir, exist_ok=True)
        for idx, row in self.data.iterrows():
            artist, song, lyrics = (
                row['Artist'], row['Song_name'], row['Lyrics']
            )

            if self.method == 'model1':
                # one 115-dim vector per song
                key = (artist, song)
                if key not in self.model1_cache:
                    self.model1_cache[key] = self.features_model1(artist, song)

            elif self.method=="model2":  # model2
                artist_key = "_".join(artist.lower().split())
                song_key   = "_".join(song.lower().split())
                cache_fname = f"{artist_key}___{song_key}.pt"
                cache_path  = os.path.join(self.model2_cache_dir, cache_fname)
                if os.path.exists(cache_path):
                    print(f"Loading cached MIDI‐features for '{artist} - {song}'")
                    self.model2_cache[idx] = torch.load(cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        artist   = row['Artist']
        song     = row['Song_name']
        lyrics   = row['Lyrics']

        # 1) text → embeddings + labels
        word_vec, labels = self.find_tokens_vec(lyrics)  # (T, E), (T,)
        T = word_vec.size(0)

        # 2) fetch precomputed MIDI features
        if self.method == 'model1':
            # broadcast the 115-dim vector to every timestep
            midi_vec = self.model1_cache[(artist, song)]    # (115,)
            midi_feats = midi_vec.unsqueeze(0).expand(T, -1)  # (T,115)

        else:  # model2
            try:
                midi_feats = self.model2_cache[idx]             # (T,17)
            
            except KeyError:
                 return None

        # 3) concat and return
        inputs = torch.cat([word_vec, midi_feats], dim=1)  # (T, E+115) or (T, E+17)
        return inputs, labels

    def filter_chars_from_lyrics(self, lyrics):
        """Removes special characters from lyrics to simplify the text."""
        return re.sub(f'[{filters}]', '', lyrics)

    def features_model1(self, artist, song_name):
        """Extracts MIDI features using the first model approach, now with tempo and drum-flag,  
        and *always* returns a 115-dim vector."""
        midi_file = self.retrieve_midi(artist, song_name)
        try:
            midi = pretty_midi.PrettyMIDI(os.path.join(self.midi_path, midi_file))

            # --- existing feature extraction ---
            beats      = midi.get_beats()                                       # (B,)
            notes_flat = np.concatenate([
                np.array([[note.start, note.end, note.pitch, note.velocity]])
                for inst in midi.instruments for note in inst.notes
            ]).flatten()                                                        # (4*N,)

            tempo      = midi.estimate_tempo()                                  # scalar
            drum_flag  = int(any(inst.is_drum for inst in midi.instruments))    # scalar

            # raw = [beats…, notes…, tempo, drum_flag]
            raw = np.concatenate([beats, notes_flat, [tempo, drum_flag]])

            # --- **pad or truncate** to fixed length 115 ---
            FIXED_LEN = 115
            if raw.shape[0] < FIXED_LEN:
                raw = np.pad(raw, (0, FIXED_LEN - raw.shape[0]), mode='constant')
            else:
                raw = raw[:FIXED_LEN]

            midi_features = torch.from_numpy(raw).float()

        except Exception:
            # fallback is also exactly 115
            midi_features = torch.zeros((115,), dtype=torch.float32)

        return midi_features

    def features_model2(self, artist, song_name, segment_idx, num_segments):
        """
        Extracts MIDI features for one specific time‐slice (segment_idx) out of num_segments:
        1) note count
        2) mean beat time
        3) average velocity
        4) average pitch
        5) drum‐flag (1 if any drum note in this segment)
        6–17) mean chroma (12 bins)
        Returns:
            Tensor of shape (17,) for the requested segment.
        """
        midi_file = self.retrieve_midi(artist, song_name)
        try:
            midi     = pretty_midi.PrettyMIDI(os.path.join(self.midi_path, midi_file))
            end_time = midi.get_end_time()
            beats    = midi.get_beats()                               # (B,)

            # Build segment boundaries
            boundaries = np.linspace(0, end_time, num_segments + 1)
            t0, t1     = boundaries[segment_idx], boundaries[segment_idx + 1]

            # 1) notes in this slice
            seg_notes = [
                (note.pitch, note.velocity)
                for inst in midi.instruments
                for note in inst.notes
                if t0 <= note.start < t1
            ]
            if seg_notes:
                pitches    = [p for p, v in seg_notes]
                velocities = [v for p, v in seg_notes]
                note_count = len(seg_notes)
                avg_pitch  = float(np.mean(pitches))
                avg_vel    = float(np.mean(velocities))
            else:
                note_count = 0
                avg_pitch  = 0.0
                avg_vel    = 0.0

            # 2) beats in this slice
            seg_beats = beats[(beats >= t0) & (beats < t1)]
            avg_beat  = float(np.mean(seg_beats)) if seg_beats.size else 0.0

            # 3) drum‐flag in this slice
            drum_flag = int(any(
                inst.is_drum and any(t0 <= note.start < t1 for note in inst.notes)
                for inst in midi.instruments
            ))

            # 4) chroma mean in this slice
            chroma       = midi.get_chroma(fs=100)                    # (12, T_frames)
            chroma_times = np.linspace(0, end_time, chroma.shape[1])
            mask         = (chroma_times >= t0) & (chroma_times < t1)
            if mask.any():
                chroma_mean = chroma[:, mask].mean(axis=1)
            else:
                chroma_mean = np.zeros(12)

            feats = [
                note_count,
                avg_beat,
                avg_vel,
                avg_pitch,
                drum_flag
            ] + chroma_mean.tolist()  # total length = 5 + 12 = 17

            return torch.tensor(feats, dtype=torch.float32)

        except Exception:
            # on error, return zero‐vector of length 17
            return torch.zeros((17,), dtype=torch.float32)



    def find_tokens_vec(self, lyrics):
        """Converts lyrics into vectors using the word2vec model and prepares labels."""
        vec, labels = [], []
        lyrics_tokens = self.filter_chars_from_lyrics(lyrics).split()
        lyrics_tokens = [word for word in lyrics_tokens if word]
        for word in lyrics_tokens:
            labels.append(self.word2index.get(word, 0))
            if word in self.word2vec:
                vec.append(self.word2vec[word])
            else:
                vec.append(np.zeros((300,)))
        vec = torch.tensor(vec, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        return vec, labels #label = teacher forcing

    def retrieve_midi(self, artist, song_name):
        """Retrieves the MIDI file corresponding to the specified artist and song name."""
        # artist = artist.lower().replace(' ', '_')
        # song_name = song_name.lower().replace(' ', '_')
        # file_name = f"{artist}_-_{song_name}.mid".replace(" ", "")
        # The Eagles, Wasted Times >> the_eagles_-_wasted_times
        # try:
        #     midi_file = next(filter(lambda x: x.lower() in file_name, os.listdir(self.midi_path)))
        #     return midi_file
        # except StopIteration:
        #     print(f"Warning: No MIDI file found for {artist}-{song_name} in {self.midi_path}")
        # return None
        # turn "Billy Joel" → "billy_joel", "Honesty" → "honesty"
        
        artist_key   = "_".join(artist.lower().split())
        song_key     = "_".join(song_name.lower().split())
        key          = f"{artist_key}_-_{song_key}"
        for fname in os.listdir(self.midi_path):
            low = fname.lower()
            if not low.endswith(('.mid','.midi')):
                continue
            if key in low:
                return fname
        print(f"Warning: No MIDI file found for {artist_key}-{song_key} in {self.midi_path}")
        return None



# Collate Function


def pad_collate(batch):
    """
    Custom collate_fn that:
      - Filters out any None (just in case)
      - Pads each `inputs` to the same max sequence length in the batch
      - Pads `labels` the same way
      - Returns a tuple (batch_inputs, batch_labels, lengths)
    
    batch: list of items returned by __getitem__, i.e. [(inputs1, labels1), (inputs2, labels2), …]
           where inputs_i has shape (Ti, D) and labels_i has shape (Ti,).
    """

    # 1) Filter out any None (though you said you used Option 1, so __getitem__ should return only valid pairs)
    filtered = [item for item in batch if item is not None]
    if len(filtered) == 0:
        # If everything got filtered out, return empty tensors
        return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    # 2) Compute the max length across the batch
    lengths = [inp.size(0) for inp, lab in filtered]
    Tmax    = max(lengths)

    # 3) Figure out the feature-dimension D from the first example
    D = filtered[0][0].size(1)   # e.g. 115+E or 17+E

    B = len(filtered)  # batch size

    # 4) Create output tensors, filled with zero (for inputs) and zero (for labels)
    #    You might choose a different pad‐value for labels (e.g. -100) if you want to ignore them in loss.
    batch_inputs = torch.zeros((B, Tmax, D), dtype=torch.float32)
    batch_labels = torch.zeros((B, Tmax), dtype=torch.long)

    # 5) Copy each (Ti, D) into the left‐most slice of (Tmax, D)
    for i, (inp, lab) in enumerate(filtered):
        Ti = inp.size(0)
        batch_inputs[i, :Ti, :] = inp
        batch_labels[i, :Ti]    = lab

    # 6) Return (padded_inputs, padded_labels, lengths)
    return batch_inputs, batch_labels, torch.tensor(lengths, dtype=torch.long)



# # Model


class LSTMLyrics(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,dropout_rate):
        super(LSTMLyrics, self).__init__()
        self.first_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.second_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.first_linear = nn.Linear(hidden_size, hidden_size)
        self.second_linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input, hidden=None, return_state=False):
        input = input.to(torch.float32)
        input = input.to(device)
        out, hidden = self.first_lstm(input, hidden)
        out, hidden = self.second_lstm(out, hidden)
        out = self.dropout(self.first_linear(out))
        logits = self.second_linear(out)
        if return_state:
            return logits, hidden
        else:
            return logits


# EarlyStopping Class


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            if self.verbose:
                print(f'New best score ({val_loss:.6f}).')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.counter = 0


# # Train Function



def compute_pairwise_metrics(pred_tokens, true_tokens, w2v_model):
    # --- Jaccard similarity over token sets ---
    set_p, set_t = set(pred_tokens), set(true_tokens)
    jaccard = len(set_p & set_t) / (len(set_p | set_t) + 1e-8)

    # --- Cosine similarity on averaged Word2Vec embeddings ---
    def avg_emb(tokens):
        vecs = [w2v_model[w] for w in tokens if w in w2v_model]
        if not vecs:
            return np.zeros(w2v_model.vector_size)
        return np.mean(vecs, axis=0)
    emb_p, emb_t = avg_emb(pred_tokens), avg_emb(true_tokens)
    cosine = cosine_similarity([emb_p], [emb_t])[0,0]

    # --- Normalized Levenshtein distance (turned into a similarity) ---
    joined_p, joined_t = " ".join(pred_tokens), " ".join(true_tokens)
    dist = edit_distance(joined_p, joined_t)
    max_len = max(len(joined_p), len(joined_t), 1)
    lev_sim = 1.0 - dist / max_len

    # --- Polarity similarity via TextBlob sentiment polarity ---
    pol_p = TextBlob(joined_p).sentiment.polarity
    pol_t = TextBlob(joined_t).sentiment.polarity
    polarity_sim = 1.0 - abs(pol_p - pol_t)

    return jaccard, cosine, lev_sim, polarity_sim


def evaluate_metrics(model, dataloader, device, idx2word, w2v_model):
    model.eval()
    scores = {"jaccard": [], "cosine": [], "lev_sim": [], "polarity": []}
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)                   # (B, T, V)
            preds = logits.argmax(dim=-1).cpu().numpy()
            trues = labels.cpu().numpy()
            for pseq, tseq in zip(preds, trues):
                # convert token IDs back to lists of words, skipping pads (0)
                p_tokens = [idx2word[i] for i in pseq if i != 0]
                t_tokens = [idx2word[i] for i in tseq if i != 0]
                j, c, l, pol = compute_pairwise_metrics(p_tokens, t_tokens, w2v_model)
                scores["jaccard"].append(j)
                scores["cosine"].append(c)
                scores["lev_sim"].append(l)
                scores["polarity"].append(pol)
    # take epoch averages
    return {k: np.mean(v) for k, v in scores.items()}


def train_model(
    train_dataloader,
    validation_dataloader,
    input_size,
    hidden_size,
    epochs,
    device,
    vocab_size,
    learning_rate,
    optimizer_name,
    index2word,               # needed to decode tokens
    w2v_model,              # your gensim word2vec loader
    project_name,
    method,
    batch_size,
    dropout_rate,
    weight_decay
):
    import wandb
    wandb.init(
    project=project_name,
    name=method,
    config={
        "method":       method,
        "batch_size":   batch_size,
        "input_size":   input_size,
        "hidden_size":  hidden_size,
        "epochs":       epochs,
        "learning_rate":learning_rate,
        "optimizer":    optimizer_name,
        "dropout_rate":  dropout_rate,
        "weight_decay": weight_decay

    }
)

    model     = LSTMLyrics(input_size, hidden_size, vocab_size,dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    early_stopping = EarlyStopping(patience=20, verbose=True)

    metrics_history = {
        "train_loss": [], "val_loss": [],
        "jaccard": [], "cosine": [], "lev_sim": [], "polarity": []
    }
   
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for inputs, labels,_ in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss   = criterion(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )
            loss.backward()
            optimizer.step()

            wandb.log({"batch/train_loss": loss.item()})
            batch_losses.append(loss.item())
            

        mean_train_loss = np.mean(batch_losses)

        # validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, labels,_ in validation_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss   = criterion(
                    logits.view(-1, vocab_size),
                    labels.view(-1)
                )
                val_losses.append(loss.item())
        mean_val_loss = np.mean(val_losses)

        # compute our new metrics on validation set
        metric_vals = evaluate_metrics(
            model, validation_dataloader, device, index2word, w2v_model
        )

        # log everything in one go
        wandb.log({
            "epoch/train_loss": mean_train_loss,
            "epoch/val_loss":   mean_val_loss,
            "epoch/jaccard":    metric_vals["jaccard"],
            "epoch/cosine":     metric_vals["cosine"],
            "epoch/lev_sim":    metric_vals["lev_sim"],
            "epoch/polarity":   metric_vals["polarity"],
        })

        # store for return
        metrics_history["train_loss"].append(mean_train_loss)
        metrics_history["val_loss"].append(mean_val_loss)
        for k in ["jaccard","cosine","lev_sim","polarity"]:
            metrics_history[k].append(metric_vals[k])

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss: {mean_train_loss:.4f}  "
            f"val_loss: {mean_val_loss:.4f}  "
            f"Jaccard: {metric_vals['jaccard']:.3f}  "
            f"Cosine: {metric_vals['cosine']:.3f}"
        )

        early_stopping(mean_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training completed in {(time.time()-t0)/60:.2f} min")
    wandb.save('model_method_2.pth')
    wandb.finish()

    return model, metrics_history


# # Create datasets and dataloaders


def prepare_dataloader(df, batch_size, method):
    """
    Prepares a DataLoader for song data, which facilitates batch processing during model training or evaluation.
    
    Parameters:
        df (DataFrame): A pandas DataFrame containing the song data.
        batch_size (int): Number of data points to load per batch.
        method (str): Specifies the method to extract MIDI features used in the SongDataset.

    Returns:
        DataLoader: A DataLoader object that provides iterable over the dataset with specified batch size and shuffling.
    """
    dataset = SongDataset(df, word2vec, word2index, vocab, './midi_files', method=method)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=pad_collate)
    return dataloader


# # First Model Train





# # Second Model Train


# 1) Prepare your data
train_dataloader_2 = prepare_dataloader(train_df, 32, method='model2')
val_dataloader_2   = prepare_dataloader(val_df,   32, method='model2')

# 3) Train model and grab metrics_history
model, metrics_history = train_model(
    train_dataloader=train_dataloader_2,
    validation_dataloader=val_dataloader_2,
    input_size=300 + 17,
    hidden_size=256,
    epochs=50,
    device=device,
    vocab_size=vocab_size,
    learning_rate=0.001,
    optimizer_name='Adam',
    index2word=index2word,        # map from token ID → word
    w2v_model=word2vec,       # your gensim Word2Vec loader
    project_name="lyrics_rnn_model2",
    method="model2_opt",
    batch_size=32,
    weight_decay=0.0,
    dropout_rate=0.5
)

# 4) Print out per-epoch metrics once training is done
n_epochs = len(metrics_history['train_loss'])
for e in range(n_epochs):
    print(
        f"Epoch {e+1}/{n_epochs}  "
        f"train_loss={metrics_history['train_loss'][e]:.4f}  "
        f"val_loss={metrics_history['val_loss'][e]:.4f}  "
        f"Jaccard={metrics_history['jaccard'][e]:.4f}  "
        f"Cosine={metrics_history['cosine'][e]:.4f}  "
        f"LevSim={metrics_history['lev_sim'][e]:.4f}  "
        f"Polarity={metrics_history['polarity'][e]:.4f}"
    )

# 5) Save & upload checkpoint
torch.save(model.state_dict(), 'model_method_2_after_opt.pth')
print("saved model model_method_2.pth" )


# # Loading Model


def load_model(model_path, input_size, hidden_size, vocab_size,dropout_rate=0.5):
    model = LSTMLyrics(input_size, hidden_size, vocab_size,dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


# # Generate Lyrics


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
from collections import Counter
def generate_song(model, first_word, midi_f, seq, train_dataset):
    output_sequence = []
    song_words = [first_word]
    hidden = None
    input_sequence = seq
    while len(output_sequence) < 200:
        logits, hidden = model(input_sequence, hidden, return_state=True)

        soft_tensor = torch.softmax(logits, dim=-1)
        next_word_idx = torch.multinomial(soft_tensor, num_samples=1).item()

        while next_word_idx == vocab_size - 1:  # To avoid padding index
            next_word_idx = torch.multinomial(soft_tensor, num_samples=1).item()

        song_words.append(train_dataset.index2word[next_word_idx])
        word_vector = train_dataset.find_tokens_vec(train_dataset.index2word[next_word_idx])[0]
        next_word = torch.cat([word_vector, midi_f], dim=1)
        input_sequence = torch.tensor(next_word, dtype=torch.float, device=device)
        hidden = tuple(h.detach() for h in hidden)  # Detach each hidden state
        output_sequence.append(next_word)
        if len(output_sequence) % 10 == 0:
            input_sequence = seq
            hidden = None

    song = "\n".join(" ".join(song_words[i:i+5]) for i in range(0, len(song_words), 5))
    return song


    print(song)


def choose_initial_word(strategy, test_lyrics=None):
    global random_initial_word
    if strategy == "random":
        if random_initial_word is None:
            random_initial_word = random.choice(list(word2index.keys()))
        return random_initial_word
    elif strategy == "most_common":
        all_words = [word for tokens in train_set['tokens'] for word in tokens]
        word_counts = Counter(all_words)
        most_common_word = word_counts.most_common(1)[0][0]
        return index2word[most_common_word]
    elif strategy == "least_common":
        all_words = [word for tokens in train_set['tokens'] for word in tokens]
        word_counts = Counter(all_words)
        least_common_word = word_counts.most_common()[-1][0]
        return index2word[least_common_word]
    elif strategy == "test_lyrics" and test_lyrics is not None:
        return test_lyrics.split()[0]
    else:
        raise ValueError("Invalid strategy. Choose from 'random', 'most_common', 'least_common', or 'test_lyrics'.")


def generate_and_collect_songs(model, model_num, midi_features, test_seqs, test_data, test_set, strategies,song_artist):
    generated_lyrics = []
    for i in range(len(test_data)):
        # pull the correct artist & song strings
        artist, song = song_artist[i]
    for i in range(len(test_data)):
        test_lyrics = test_set.iloc[i]['Lyrics']
        for strategy in strategies:
            initial_word = choose_initial_word(strategy, test_lyrics)
            if midi_features[i] is None or test_seqs[i] is None:
                continue
            generated_song = generate_song(model, initial_word, midi_features[i], test_seqs[i], test_data)
            
            generated_lyrics.append((song, artist, model_num, strategy, initial_word, generated_song))
    return generated_lyrics


def collect_and_print_all_songs(method,model, model_num, midi_features, test_seqs, test_data, test_set, strategies):
    all_generated_lyrics = []
    song_artist =list(zip(test_set["Artist"], test_set["Song_name"]))
    # Generate Test Songs for Method 1
    lyrics_approach = generate_and_collect_songs(model, model_num, midi_features, test_seqs, test_data, test_set, strategies,song_artist)
    all_generated_lyrics.extend(lyrics_approach)

    return all_generated_lyrics

# Define sentiment comparison function
def analyze_sentiment_tone(word: str, lyrics: str) -> float:
    analyzer = SentimentIntensityAnalyzer()
    word_score = analyzer.polarity_scores(word)['compound']
    song_score = analyzer.polarity_scores(lyrics)['compound']
    print(f"  Sentiment score for word:  {word_score:.3f}")
    print(f"  Sentiment score for song:  {song_score:.3f}")
    similarity = (1 - abs(word_score - song_score) / 2) * 100
    print(f"  Sentiment similarity:      {similarity:.1f}%")
    return similarity

# Analyze lyrics from a DataFrame or CSV
def analyze_effects(generated_lyrics):
    analysis = []
    for strategy in strategies:
        lyrics_with_strategy = [lyrics for lyrics in generated_lyrics if lyrics[3] == strategy]
        analysis.append((strategy, lyrics_with_strategy))

    for strategy, lyrics in analysis:
        print(f"\nAnalysis for strategy: {strategy}")
        for song in lyrics:
            if song[0]=="Song_name": continue
            print(f"\nSong: {song[0]} by {song[1]}")
            print(f"Model: {song[2]}")
            print(f"Initial Word: {song[4]}")
            print(f"Generated Lyrics:\n{song[5]}\n")
            print(f"song[4]: {song[4]}, song[5]: {song[5]}")
            analyze_sentiment_tone(song[4], song[5])
        print("--------------------------------\n")
       
        


# ### Generate for model 1


# Set a fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Store the randomly chosen word
random_initial_word = None

input_size_model1 = 300 + 115  
hidden_size = 256  

first_model = load_model("./model_method_1_after_opt.pth", input_size_model1, hidden_size, vocab_size)

# Create the Test DataLoader for both approaches
test_dataset_model1 = SongDataset(test_set, word2vec, vocab, './midi_files/', method='model1')
test_dataloader_model1 = DataLoader(test_dataset_model1, batch_size=1, shuffle=True)

# Prepare sequences and features for testing
test_seq_model1 = [lyrics[:, 0, :] for lyrics, _ in test_dataloader_model1]


test_data = [(artist, song) for artist, song in zip(test_set['Artist'], test_set['Song_name'])]
midi_features_model1=[]

for artist, song_name in test_data:
    midi_features_model1.append(test_dataset_model1.features_model1(artist, song_name).view(1, 115))

# Define the strategies
strategies = ["random", "most_common", "least_common", "test_lyrics"]

# Generating and collecting lyrics for different initial word strategies
all_generated_lyrics = collect_and_print_all_songs("model1")

generated_lyrics_df = pd.DataFrame(all_generated_lyrics, columns=["Song Name", "Artist", "Model", "Strategy", "Initial Word", "Generated Lyrics"])
generated_lyrics_df.to_csv("generated_lyrics_report.csv", index=False)
  
print("Analyzing effects of the initial word and melody on the generated lyrics:")
analyze_effects(all_generated_lyrics)





# ### Generate for model2


# Set a fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Store the randomly chosen word
random_initial_word = None


input_size_model2 = 300 + 17  
hidden_size = 256  
second_model = load_model("./model_method_2_after_opt.pth", input_size_model2, hidden_size, vocab_size,dropout_rate=0.5)
test_dataset_model2 = SongDataset(
    df       = test_set,
    word2vec = word2vec,
    word2index=word2index,
    vocab      = vocab,              # correct object here
    midi_path  = './midi_files/',    # proper string path
    method     = 'model2'
)
# test_dataset_model2 = SongDataset(test_set, word2vec,word2index, vocab, './midi_files/', method='model2')
test_dataloader_model2 = DataLoader(test_dataset_model2, batch_size=1, shuffle=True,collate_fn=pad_collate)


N = len(test_set)

# Make all three containers exactly N long
midi_features_model2 = [None] * N
test_seq_model2      = [None] * N
test_data            = list(zip(test_set["Artist"], test_set["Song_name"]))

for idx, row in test_set.reset_index(drop=True).iterrows():
    artist, song = row["Artist"], row["Song_name"]
    print(artist,song)

    # ----- MIDI features -------------------------------------------------
    cache_fname = f"{artist.lower().replace(' ','_')}___{song.lower().replace(' ','_')}.pt"
    cache_path  = os.path.join("./midi_files/model2_cache", cache_fname)
    if os.path.exists(cache_path):
        midi_features_model2[idx] = torch.load(cache_path)          # (17,)
    else:                           # fallback so we keep the slot
        print("no midi features for this")
        midi_features_model2[idx] = torch.zeros(17)

    # ----- input sequence ------------------------------------------------
    sample = test_dataset_model2[idx]    
    
    if sample is None:
        # keep the slot so all lists stay the same length
        test_seq_model2[idx] = None
    else:
        seq_tensor, _ = sample              # normal case
        test_seq_model2[idx] = seq_tensor
        


# Define the strategies
strategies = ["random", "most_common", "least_common", "test_lyrics"]

# Generating and collecting lyrics for different initial word strategies
all_generated_lyrics = collect_and_print_all_songs("model2",second_model,2,midi_features_model2,test_seq_model2,test_dataset_model2,test_set,strategies)

# Save all generated lyrics for the report


generated_lyrics_df = pd.DataFrame(all_generated_lyrics, columns=["Song Name", "Artist", "Model", "Strategy", "Initial Word", "Generated Lyrics"])
generated_lyrics_df.to_csv("generated_lyrics_report.csv", index=False)
  
print("Analyzing effects of the initial word and melody on the generated lyrics:")
analyze_effects(all_generated_lyrics)


#### Optuna runs

def objective_model2(trial):
    # 1) Suggest hyperparameters
    batch_size    = trial.suggest_categorical("batch_size",    [32, 64])
    hidden_size   = trial.suggest_categorical("hidden_size",   [128, 256])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3])
    dropout_rate  = trial.suggest_categorical("dropout_rate",  [0.2, 0.3, 0.5])


    # 2) Build new DataLoaders for this batch size
    train_loader = prepare_dataloader(train_df, batch_size, method='model2')
    val_loader   = prepare_dataloader(val_df,   batch_size, method='model2')

    # 3) Train for a small number of epochs (e.g. 5) to get a quick val loss
    _, history = train_model(
        train_loader, val_loader,
        input_size=300+17,
        hidden_size=hidden_size,
        epochs=50,
        device=device,
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        optimizer_name='Adam',
        index2word=index2word,
        w2v_model=word2vec,
        project_name="optuna_lyrics_model2",
        method="model2",
        batch_size=batch_size,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate
    )

    # 4) Return the last epoch's validation loss
    return history['val_loss'][-1]

import optuna
# 5) Launch the study
study = optuna.create_study(direction="minimize")
study.optimize(objective_model2, n_trials=50)

print("Best hyperparameters found:")
for k,v in study.best_params.items():
    print(f"  {k}: {v}")


def objective_model1(trial):
    # 1) Suggest hyperparameters
    batch_size    = trial.suggest_categorical("batch_size",    [32, 64])
    hidden_size   = trial.suggest_categorical("hidden_size",   [128, 256])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3])
    dropout_rate  = trial.suggest_categorical("dropout_rate",  [0.2, 0.3, 0.5])


    # 2) Build new DataLoaders for this batch size
    train_loader = prepare_dataloader(train_df, batch_size, method='model1')
    val_loader   = prepare_dataloader(val_df,   batch_size, method='model1')

    # 3) Train for a small number of epochs (e.g. 5) to get a quick val loss
    _, history = train_model(
        train_loader, val_loader,
        input_size=300+115,
        hidden_size=hidden_size,
        epochs=50,
        device=device,
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        optimizer_name='Adam',
        index2word=index2word,
        w2v_model=word2vec,
        project_name="optuna_lyrics_model1",
        method="model1",
        batch_size=batch_size,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate
    )

    # 4) Return the last epoch's validation loss
    return history['val_loss'][-1]

# 5) Launch the study
study = optuna.create_study(direction="minimize")
study.optimize(objective_model1, n_trials=50)

print("Best hyperparameters found:")
for k,v in study.best_params.items():
    print(f"  {k}: {v}")


