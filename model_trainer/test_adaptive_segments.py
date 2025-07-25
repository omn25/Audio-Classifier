import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from model import SongModel
from utils import segment_for_game_adaptive, segment_for_game_smart
import os
from pydub import AudioSegment
import argparse

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

def analyze_song_adaptive(song_path, model, scaler=None, method='adaptive', top_k=3, 
                         threshold=0.25, min_len=5000, max_len=10000):
    """Analyze a song using adaptive segmentation"""
    print(f"\n🎵 Analyzing: {os.path.basename(song_path)}")
    print(f"📊 Method: {method}")
    
    # Load the original song
    original_song = AudioSegment.from_file(song_path)
    song_name = os.path.splitext(os.path.basename(song_path))[0]
    
    if method == 'adaptive':
        # Test multiple segment lengths and let model choose (5s-10s)
        segment_lengths = [5000, 8000, 10000]  # 5s, 8s, 10s
        segments = segment_for_game_adaptive(
            song_path, song_name, model, scaler, 
            segment_lengths=segment_lengths, 
            top_k=top_k
        )
    elif method == 'smart':
        # Smart expansion with configurable parameters
        segments = segment_for_game_smart(
            song_path, song_name, model, scaler,
            min_len=min_len, max_len=max_len, threshold=threshold
        )
    else:
        print(f"Unknown method: {method}")
        return []
    
    if not segments:
        print("❌ No good segments found!")
        return []
    
    # Extract and save segments
    extracted_segments = []
    os.makedirs("game_segments", exist_ok=True)
    
    print(f"\n🎯 Found {len(segments)} segments:")
    for i, segment in enumerate(segments[:top_k]):
        song_name, start_ms, end_ms, *features = segment
        
        # Calculate duration
        duration = (end_ms - start_ms) / 1000
        
        # Convert to time format
        start_time = f"{start_ms//1000//60}:{(start_ms//1000)%60:02d}"
        end_time = f"{end_ms//1000//60}:{(end_ms//1000)%60:02d}"
        
        # Extract audio segment
        segment_audio = original_song[start_ms:end_ms]
        
        # Create filename
        segment_filename = f"game_segments/{song_name}_adaptive_{i+1:02d}_{start_time.replace(':', '-')}_{end_time.replace(':', '-')}.wav"
        
        # Export as WAV
        segment_audio.export(segment_filename, format="wav")
        
        # Get prediction score
        features = np.array(features[:27])
        if scaler:
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        with torch.no_grad():
            prediction = model(torch.FloatTensor(features_scaled))
            score = prediction.item()
        
        segment_info = {
            'song_name': song_name,
            'start_time': start_time,
            'end_time': end_time,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'duration': duration,
            'score': score,
            'filename': segment_filename
        }
        
        extracted_segments.append(segment_info)
        
        print(f"  {i+1}. {start_time} - {end_time} "
              f"(Duration: {duration:.1f}s, Score: {score:.3f})")
        print(f"     📁 Saved as: {segment_filename}")
    
    return extracted_segments

def main():
    """Main function to test adaptive segmentation"""
    parser = argparse.ArgumentParser(description='Test adaptive segment length selection')
    parser.add_argument('--method', choices=['adaptive', 'smart'], default='adaptive',
                       help='Segmentation method (default: adaptive)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top segments to extract (default: 3)')
    parser.add_argument('--songs-dir', type=str, default='test_songs',
                       help='Directory containing songs to analyze (default: test_songs)')
    parser.add_argument('--threshold', type=float, default=0.25,
                       help='Score threshold for smart method (default: 0.25)')
    parser.add_argument('--min-len', type=int, default=5000,
                       help='Minimum segment length in ms (default: 5000)')
    parser.add_argument('--max-len', type=int, default=10000,
                       help='Maximum segment length in ms (default: 10000)')
    
    args = parser.parse_args()
    
    # Load trained model
    print("🤖 Loading model...")
    model, scaler = load_trained_model()
    if model is None:
        return
    
    print("✅ Model loaded successfully!")
    print(f"🔧 Scaler available: {scaler is not None}")
    
    # Test on songs
    songs_dir = args.songs_dir
    if not os.path.exists(songs_dir):
        print(f"❌ Songs directory {songs_dir} not found!")
        return
    
    song_files = [f for f in os.listdir(songs_dir) if f.endswith('.mp3')]
    if not song_files:
        print("❌ No MP3 files found!")
        return
    
    print(f"\n📁 Found {len(song_files)} songs to analyze:")
    for song in song_files:
        print(f"  - {song}")
    
    # Analyze each song
    all_results = {}
    for song_file in song_files:
        song_path = os.path.join(songs_dir, song_file)
        results = analyze_song_adaptive(song_path, model, scaler, 
                                      method=args.method, top_k=args.top_k,
                                      threshold=args.threshold, min_len=args.min_len, max_len=args.max_len)
        
        if results:
            all_results[song_file] = results
    
    # Save results
    if all_results:
        save_results_to_csv(all_results, f'adaptive_segments_{args.method}.csv')
        print(f"\n💾 Results saved to adaptive_segments_{args.method}.csv")
        print(f"🎵 WAV files saved to game_segments/ directory")

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
                'duration': segment['duration'],
                'score': segment['score'],
                'filename': segment['filename']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"📊 Saved {len(rows)} segments to {filename}")

if __name__ == "__main__":
    main() 