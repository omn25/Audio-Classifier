import pandas as pd
from utils import segment_and_label, time_to_ms, extract_features
import os
from pydub import AudioSegment
import numpy as np

labels = {
    "maahi_ve": [("1:55", "2:55"), ("4:02", "4:07"), ("5:25", "6:00")],
    "tera_hone_laga_hoon": [("2:47", "2:57")],
    "soch_na_sake": [("0:01", "0:05")],
    "jeene_laga_hoon": [("0:14", "0:18"), ("0:32", "0:50")],
    "naina": [("0:02", "0:24"), ("1:25", "1:35"), ("1:35", "1:45")],
    "pehla_nasha": [("0:00", "0:25"), ("0:50", "1:00"), ("0:50", "1:10"), ("2:00", "2:08")],
    "ye_ishq_hai": [("2:20", "2:37")],
    "abhi_mujhe_mein_kahin": [("0:25", "0:31"), ("1:37", "1:42")],
    "phir_bhi_tumko_chaahunga": [("0:00", "0:10")],
    "suraj_hua_maddham": [("3:47", "3:52")],
    "kabira": [("0:10", "0:29"), ("1:45", "1:57")],
    "illahi": [("0:00", "0:10"), ("2:12", "2:35")],
    "dagabaaz_re": [("0:12", "0:16"), ("1:00", "1:20"), ("2:18", "2:25")],
    "kisi_disco_mein_jaaye": [("0:00", "0:10"), ("3:10", "3:45")],
    "jaane_nahin_denge_tujhe": [("1:07", "1:10"), ("2:05", "2:10")],
    "kalank": [("0:00", "0:14"), ("0:48", "0:55"), ("1:46", "2:16"), ("3:48", "3:55")]
}

all_segments = []

for song, ranges in labels.items():
    path = f"songs/{song}.mp3"

    # Convert all ranges to milliseconds
    ms_ranges = [(time_to_ms(start), time_to_ms(end)) for (start, end) in ranges]

    # Use all good segments to label
    def is_good_segment(start, end):
        return any(not (end <= s or start >= e) for (s, e) in ms_ranges)

    song_audio = AudioSegment.from_file(path)
    segment_len = 5000
    step_size = 2500

    for i in range(0, len(song_audio) - segment_len, step_size):
        segment = song_audio[i:i + segment_len]
        y = np.array(segment.get_array_of_samples()).astype(np.float32)
        sr = segment.frame_rate
        features = extract_features(y, sr)
        label = int(is_good_segment(i, i + segment_len))
        all_segments.append([song, i, i + segment_len, label] + features.tolist())

# Save to CSV
columns = ["song", "start_ms", "end_ms", "label"] + [f"feat_{i}" for i in range(len(all_segments[0]) - 4)]
df = pd.DataFrame(all_segments, columns=columns)
df.to_csv("segments.csv", index=False)

print("✅ Dataset saved with updated segments!")