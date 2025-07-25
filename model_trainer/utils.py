import librosa
import numpy as np
from pydub import AudioSegment
import os
import torch

def time_to_ms(tstr):
    """Convert mm:ss to milliseconds"""
    m, s = map(int, tstr.split(":"))
    return (m * 60 + s) * 1000

def overlaps(seg_start, seg_end, ideal_start, ideal_end):
    """Check if segment overlaps with the target"""
    return not (seg_end <= ideal_start or seg_start >= ideal_end)

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    rms = librosa.feature.rms(y=y).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    return np.concatenate([mfcc, chroma, [rms, spectral_centroid]])

def extract_game_features(y, sr):
    """Extract features specifically for game segment analysis"""
    # Use the same features as training data for compatibility
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    rms = librosa.feature.rms(y=y).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    return np.concatenate([mfcc, chroma, [rms, spectral_centroid]])

def segment_and_label(file_path, song_name, ideal_start_ms, ideal_end_ms, segment_len=5000, step_size=2500):
    song = AudioSegment.from_file(file_path)
    segments = []

    for i in range(0, len(song) - segment_len, step_size):
        segment = song[i:i+segment_len]
        y = np.array(segment.get_array_of_samples()).astype(np.float32)
        sr = segment.frame_rate
        features = extract_features(y, sr)

        label = int(overlaps(i, i+segment_len, ideal_start_ms, ideal_end_ms))
        segments.append([song_name, i, i+segment_len, label] + features.tolist())

    return segments

def segment_for_game(file_path, song_name, segment_len=5000, step_size=2500):
    """Segment a song for game analysis without labels"""
    song = AudioSegment.from_file(file_path)
    segments = []

    for i in range(0, len(song) - segment_len, step_size):
        segment = song[i:i+segment_len]
        y = np.array(segment.get_array_of_samples()).astype(np.float32)
        sr = segment.frame_rate
        features = extract_game_features(y, sr)

        segments.append([song_name, i, i+segment_len] + features.tolist())

    return segments

def predict_segment_quality(model, features, scaler=None):
    """Use trained model to predict if a segment is good for the game"""
    if scaler:
        features = scaler.transform(features.reshape(1, -1))
    
    with torch.no_grad():
        prediction = model(torch.FloatTensor(features))
        return prediction.item()

def segment_for_game_adaptive(file_path, song_name, model, scaler=None, 
                             segment_lengths=[5000, 8000, 10000], 
                             step_size=2500, top_k=3):
    """
    Segment a song adaptively - let the model choose the best segment length
    for each part of the song by testing multiple lengths.
    
    Args:
        file_path: Path to the song file
        song_name: Name of the song
        model: Trained model for prediction
        scaler: Feature scaler (optional)
        segment_lengths: List of segment lengths to try (in milliseconds) - max 10s
        step_size: Step size between analysis windows
        top_k: Number of best segments to return
    """
    song = AudioSegment.from_file(file_path)
    all_candidates = []
    
    print(f"Testing {len(segment_lengths)} different segment lengths: {[f'{l/1000:.1f}s' for l in segment_lengths]}")
    
    # Test each segment length
    for segment_len in segment_lengths:
        print(f"  Analyzing with {segment_len}ms segments...")
        
        for i in range(0, len(song) - segment_len, step_size):
            segment = song[i:i+segment_len]
            y = np.array(segment.get_array_of_samples()).astype(np.float32)
            sr = segment.frame_rate
            features = extract_game_features(y, sr)
            
            # Get model prediction
            if scaler:
                features_scaled = scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            with torch.no_grad():
                prediction = model(torch.FloatTensor(features_scaled))
                score = prediction.item()
            
            all_candidates.append({
                'song_name': song_name,
                'start_ms': i,
                'end_ms': i + segment_len,
                'segment_len': segment_len,
                'score': score,
                'features': features
            })
    
    # Sort by score and remove overlapping segments
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter out overlapping segments (keep the best one)
    selected_segments = []
    for candidate in all_candidates:
        # Check if this segment overlaps with any already selected
        overlaps = False
        for selected in selected_segments:
            if (candidate['start_ms'] < selected['end_ms'] and 
                candidate['end_ms'] > selected['start_ms']):
                overlaps = True
                break
        
        if not overlaps:
            selected_segments.append(candidate)
            if len(selected_segments) >= top_k:
                break
    
    # Convert to the same format as original function
    segments = []
    for seg in selected_segments:
        segments.append([
            seg['song_name'], 
            seg['start_ms'], 
            seg['end_ms']
        ] + seg['features'].tolist())
    
    return segments

def segment_for_game_smart(file_path, song_name, model, scaler=None, 
                          min_len=5000, max_len=10000, step_size=2500, threshold=0.25):
    """
    Smart segmentation that starts with short segments and expands them
    if the model predicts they're good, until reaching optimal length (max 10s).
    """
    song = AudioSegment.from_file(file_path)
    segments = []
    tested_positions = 0
    kept_segments = 0
    
    print(f"Smart segmentation: testing lengths from {min_len/1000:.1f}s to {max_len/1000:.1f}s")
    print(f"Score threshold: {threshold}")
    
    for i in range(0, len(song) - max_len, step_size):
        tested_positions += 1
        best_score = 0
        best_len = min_len
        
        # Try expanding the segment length in larger increments (closer to training data)
        for test_len in range(min_len, max_len + 1000, 1000):  # 1000ms increments (5s, 6s, 7s, 8s, 9s, 10s)
            if i + test_len > len(song):
                break
                
            segment = song[i:i+test_len]
            y = np.array(segment.get_array_of_samples()).astype(np.float32)
            sr = segment.frame_rate
            features = extract_game_features(y, sr)
            
            # Get model prediction
            if scaler:
                features_scaled = scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            with torch.no_grad():
                prediction = model(torch.FloatTensor(features_scaled))
                score = prediction.item()
            
            # If score improves, keep expanding
            if score > best_score:
                best_score = score
                best_len = test_len
            # If score drops significantly, stop expanding
            elif score < best_score * 0.7:  # Less strict drop threshold
                break
        
        # Keep segments with scores above threshold
        if best_score > threshold:
            kept_segments += 1
            segment = song[i:i+best_len]
            y = np.array(segment.get_array_of_samples()).astype(np.float32)
            sr = segment.frame_rate
            features = extract_game_features(y, sr)
            
            segments.append([song_name, i, i+best_len] + features.tolist())
            
            # Debug info for first few segments
            if kept_segments <= 3:
                start_time = f"{i//1000//60}:{(i//1000)%60:02d}"
                end_time = f"{(i+best_len)//1000//60}:{((i+best_len)//1000)%60:02d}"
                print(f"  Position {start_time}: {best_len/1000:.1f}s segment (score: {best_score:.3f})")
    
    print(f"Smart segmentation complete: {kept_segments}/{tested_positions} positions kept")
    return segments
