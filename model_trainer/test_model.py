import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from model import SongModel
from utils import segment_for_game, extract_features
import os
from pydub import AudioSegment

def load_trained_model(model_path='best_song_model.pth'):
    """Load the trained model and scaler"""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Load model
    model = SongModel(input_size=27)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get scaler from checkpoint
    scaler = checkpoint.get('scaler')
    
    return model, scaler

def analyze_new_song(song_path, model, scaler=None, top_k=5):
    """Analyze a new song and find the best segments for the guessing game"""
    print(f"Analyzing song: {os.path.basename(song_path)}")
    
    # Load the original song
    original_song = AudioSegment.from_file(song_path)
    
    # Segment the song
    song_name = os.path.splitext(os.path.basename(song_path))[0]
    segments = segment_for_game(song_path, song_name)
    
    if not segments:
        print("No segments found!")
        return []
    
    # Extract features for prediction
    predictions = []
    for segment in segments:
        song_name, start_ms, end_ms, *features = segment
        
        # Convert to numpy array
        features = np.array(features[:27])  # Take first 27 features to match model
        
        # Scale features if scaler is available
        if scaler:
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(torch.FloatTensor(features_scaled))
            score = prediction.item()
        
        # Convert milliseconds to mm:ss format
        start_time = f"{start_ms//1000//60}:{(start_ms//1000)%60:02d}"
        end_time = f"{end_ms//1000//60}:{(end_ms//1000)%60:02d}"
        
        predictions.append({
            'song_name': song_name,
            'start_time': start_time,
            'end_time': end_time,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'score': score,
            'duration': (end_ms - start_ms) / 1000
        })
    
    # Sort by score (highest first)
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Extract top k segments as WAV files
    top_segments = predictions[:top_k]
    extracted_segments = []
    
    for i, segment in enumerate(top_segments):
        # Extract the segment from the original song
        segment_audio = original_song[segment['start_ms']:segment['end_ms']]
        
        # Create filename
        segment_filename = f"game_segments/{segment['song_name']}_segment_{i+1:02d}_{segment['start_time'].replace(':', '-')}_{segment['end_time'].replace(':', '-')}.wav"
        
        # Ensure directory exists
        os.makedirs("game_segments", exist_ok=True)
        
        # Export as WAV
        segment_audio.export(segment_filename, format="wav")
        
        # Add filename to segment info
        segment['filename'] = segment_filename
        extracted_segments.append(segment)
        
        print(f"  {i+1}. {segment['start_time']} - {segment['end_time']} "
              f"(Score: {segment['score']:.3f}, Duration: {segment['duration']:.1f}s)")
        print(f"     Saved as: {segment_filename}")
    
    return extracted_segments

def main():
    """Main function to test model on new songs"""
    # Load trained model
    model, scaler = load_trained_model()
    if model is None:
        return
    
    print("Model loaded successfully!")
    print(f"Scaler available: {scaler is not None}")
    
    # Test on songs in the songs directory
    songs_dir = "test_songs"
    if not os.path.exists(songs_dir):
        print(f"Songs directory {songs_dir} not found!")
        return
    
    # Get all mp3 files
    song_files = [f for f in os.listdir(songs_dir) if f.endswith('.mp3')]
    
    if not song_files:
        print("No MP3 files found in songs directory!")
        return
    
    print(f"\nFound {len(song_files)} songs to analyze:")
    for song in song_files:
        print(f"  - {song}")
    
    # Analyze each song
    all_results = {}
    for song_file in song_files:
        song_path = os.path.join(songs_dir, song_file)
        results = analyze_new_song(song_path, model, scaler, top_k=3)
        
        if results:
            all_results[song_file] = results
            print(f"\nTop 3 segments extracted for {song_file}")
    
    # Save results to CSV
    if all_results:
        save_results_to_csv(all_results, 'game_segments.csv')
        print(f"\nResults saved to game_segments.csv")
        print(f"WAV files saved to game_segments/ directory")

def save_results_to_csv(all_results, filename):
    """Save analysis results to CSV file"""
    rows = []
    for song_name, segments in all_results.items():
        for segment in segments:
            rows.append({
                'song': song_name,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'start_ms': segment['start_ms'],
                'end_ms': segment['end_ms'],
                'score': segment['score'],
                'duration': segment['duration'],
                'filename': segment['filename']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved {len(rows)} segments to {filename}")

if __name__ == "__main__":
    main()