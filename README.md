# 🎵 Audio Classifier

A binary classification model that labels 10-second audio segments from full songs as either **"good" (1)** or **"bad" (0)** for use in audio-based applications like guessing games, highlight extraction, or clip curation.

---

## 🔍 Overview

This project powers automated clip selection for [Bollyguess](https://www.bollyguess.ca), improving the quality of daily audio snippets by predicting which segments are likely to be familiar yet challenging.

The model is trained on manually labeled segments and extracts audio features such as MFCC, chroma, tempo, and pitch using `Librosa`. It uses a feedforward neural network built in `PyTorch`, and is evaluated with standard classification metrics.

---

## 🛠 Tech Stack

- **Python**
- **PyTorch** – model architecture and training
- **TensorFlow** – experimentation support
- **Librosa** – feature extraction (MFCC, chroma, tempo, pitch)
- **FFmpeg** – audio slicing and preprocessing
- **Scikit-learn** – evaluation (confusion matrix, F1-score)
- **Pandas**

---

## 🧠 Model Architecture

- 4 hidden layers with ReLU activation
- Final layer uses Sigmoid for binary classification
- Trained on labeled 10-second audio clips (0 = not suitable, 1 = suitable)

---

## ⚙️ Pipeline

1. **Audio Preprocessing**  
   Full songs are sliced into 10-second segments using `FFmpeg`.

2. **Feature Extraction**  
   Each segment is converted to a feature vector using `Librosa`:
   - MFCCs
   - Chroma
   - Tempo
   - Pitch

3. **Model Training**  
   A binary classifier is trained on the extracted features using `PyTorch`.

4. **Evaluation**  
   Model performance is analyzed using:
   - F1-score
   - Confusion matrix

5. **Deployment (Optional)**  
   The classifier can be integrated into apps for automated segment selection.

---

## 📊 Example Use Case

Used in [Bollyguess](https://www.bollyguess.ca) to select ideal audio clips for a daily Bollywood music guessing game.

---

## 🚀 Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/audio-classifier.git
   cd audio-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run preprocessing**
   ```bash
   python scripts/slice_audio.py --input songs/ --output clips/
   ```

4. **Extract features**
   ```bash
   python scripts/extract_features.py --input clips/ --output features.csv
   ```

5. **Train the model**
   ```bash
   python train.py --features features.csv
   ```

---

## 🧪 Sample Output

- Segment: `00:50–01:00` → **Predicted: 1**
- Segment: `02:15–02:25` → **Predicted: 0**

---

## 📂 Project Structure

```
audio-classifier/
│
├── data/
│   └── clips/                # 10-second audio segments
├── scripts/
│   ├── slice_audio.py        # FFmpeg slicing
│   ├── extract_features.py   # Feature engineering
│
├── model/
│   ├── train.py              # Training script
│   └── model.pt              # Trained weights
```

---

## 📄 License

MIT License

---

## 🙋‍♂️ Contact

Built by Om Nathwani  
Email: ornathwa@uwaterloo.ca  
GitHub: [omn25](https://github.com/omn25)

