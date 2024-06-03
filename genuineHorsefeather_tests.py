import os
import numpy as np
import pyaudio
import librosa
from sklearn.neighbors import KDTree
import joblib
import time

# Define constants
chunk_size = 1024  # Adjusted buffer size
hop_ratio = 0.5
n_mfcc = 13
n_fft = 1024  # Adjust this to be smaller for better time resolution and pitch capture
n_mels = 40  # Reduce the number of Mel filters to reduce computational load
sr = 44100
overlap_size = chunk_size // 2

# Define noise gate thresholds
threshold_on = 0.02
threshold_off = 0.01
smoothing_factor = 0.01

def extract_features(y, sr, n_mfcc, hop_length, n_fft, n_mels):
    # Compute the MFCC features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels), axis=1)

    # Calculate the fundamental frequency (F0)
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    fmin = librosa.note_to_hz('C3')
    fmax = librosa.note_to_hz('C6')
    bins_per_octave = 12
    pitch_bins = librosa.cqt_frequencies(n_bins=n_mels, fmin=fmin, bins_per_octave=bins_per_octave)
    f0 = np.zeros(D.shape[1])
    for i in range(D.shape[1]):
        frame = D[:, i]
        pitch_range_indices = np.where((pitch_bins >= fmin) & (pitch_bins <= fmax))
        peak_bin = pitch_range_indices[0][np.argmax(frame[pitch_range_indices])]
        f0[i] = librosa.hz_to_midi(pitch_bins[peak_bin])
    
    # Compute Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft), axis=1)
    
    # Compute Spectral Roll-off
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft), axis=1)

    # Combine MFCC, F0, Spectral Centroid, and Spectral Roll-off features
    features = np.concatenate([mfcc, [np.mean(f0)], spectral_centroid, spectral_rolloff])

    return features

def build_kd_tree(corpus_folder, sr):
    file_paths = []
    features = []
    offsets = []

    for root, _, files in os.walk(corpus_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                start_time = time.time()
                y, _ = librosa.load(file_path, sr=sr)
                hop_length = int(len(y) * hop_ratio)
                
                for i in range(0, len(y) - chunk_size, overlap_size):
                    chunk = y[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        continue
                    chunk_windowed = chunk * np.hanning(len(chunk))
                    feature_vector = extract_features(chunk_windowed, sr, n_mfcc, hop_length, n_fft, n_mels)
                    features.append(feature_vector)
                    file_paths.append(file_path)
                    offsets.append(i / sr)  # Store the offset in seconds

                end_time = time.time()
                print(f"Processed {file_path} in {end_time - start_time:.2f} seconds")

    kd_tree = KDTree(np.array(features))
    return kd_tree, file_paths, offsets

def save_kd_tree(corpus_folder, kd_tree, file_paths, offsets):
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')
    file_paths_file = os.path.join(corpus_folder, 'file_paths.pkl')
    offsets_file = os.path.join(corpus_folder, 'offsets.pkl')
    
    joblib.dump(kd_tree, kd_tree_file)
    joblib.dump(file_paths, file_paths_file)
    joblib.dump(offsets, offsets_file)

def load_kd_tree(corpus_folder):
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')
    file_paths_file = os.path.join(corpus_folder, 'file_paths.pkl')
    offsets_file = os.path.join(corpus_folder, 'offsets.pkl')
    
    kd_tree = joblib.load(kd_tree_file)
    file_paths = joblib.load(file_paths_file)
    offsets = joblib.load(offsets_file)
    
    return kd_tree, file_paths, offsets

def find_nearest_neighbor(feature_vector, kd_tree, file_paths, offsets):
    try:
        dist, ind = kd_tree.query([feature_vector], k=1)  # Set k=1 to find only the closest neighbor
        nearest_neighbor = (file_paths[ind[0][0]], offsets[ind[0][0]])
        return nearest_neighbor
    except Exception as e:
        print(f"Error finding nearest neighbor: {e}")
        return None

def apply_envelope(audio, envelope):
    return audio * envelope

def main():
    corpus_folder = input("Please provide the path to the folder containing WAV files: ")

    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')
    file_paths_file = os.path.join(corpus_folder, 'file_paths.pkl')
    offsets_file = os.path.join(corpus_folder, 'offsets.pkl')

    if os.path.exists(kd_tree_file) and os.path.exists(file_paths_file) and os.path.exists(offsets_file):
        print("Loading KD-tree from files...")
        kd_tree, file_paths, offsets = load_kd_tree(corpus_folder)
        print("KD-tree loaded successfully.")
    else:
        print("Building KD-tree...")
        kd_tree, file_paths, offsets = build_kd_tree(corpus_folder, sr)
        save_kd_tree(corpus_folder, kd_tree, file_paths, offsets)
        print("KD-tree built and saved successfully.")

    p = pyaudio.PyAudio()

    # Print available devices for debugging
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']} (Input: {info['maxInputChannels']}, Output: {info['maxOutputChannels']})")

    # Prompt user to input device indices
    input_device_index = int(input("Enter the input device index: "))
    output_device_index = int(input("Enter the output device index: "))

    # Open input stream for recording from microphone
    input_stream = p.open(format=pyaudio.paFloat32,
                          channels=1,
                          rate=sr,
                          input=True,
                          input_device_index=input_device_index,
                          frames_per_buffer=chunk_size)

    # Open output stream for playback
    output_stream = p.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=sr,
                           output=True,
                           output_device_index=output_device_index)

    # Initialize gate state
    gate_open = False
    fade_samples = chunk_size // 10  # Number of samples to use for fade-in/fade-out
    fade_curve = np.linspace(0, 1, fade_samples)

    previous_rms = 0.0

    try:
        previous_chunk = np.zeros(chunk_size, dtype=np.float32)
        window = np.hanning(chunk_size)
        current_audio_chunk = None
        while True:
            try:
                # Read chunk from microphone
                data = input_stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Calculate the RMS of the input audio chunk
                rms = np.sqrt(np.mean(np.square(audio_chunk)))

                # Smooth the RMS value
                rms = smoothing_factor * previous_rms + (1 - smoothing_factor) * rms
                previous_rms = rms

                envelope = rms
                max_envelope = np.max(envelope)
                envelope *= 3.0
                
                # Apply noise gate with hysteresis
                if gate_open:
                    if max_envelope < threshold_off:
                        gate_open = False
                else:
                    if max_envelope > threshold_on:
                        gate_open = True

                if gate_open:
                    if current_audio_chunk is None or (current_audio_chunk is not None and len(current_audio_chunk) < chunk_size):
                        # Apply Hann window to the input audio chunk
                        audio_chunk_windowed = audio_chunk * window

                        # Extract feature vector for the chunk
                        feature_vector = extract_features(audio_chunk_windowed, sr, n_mfcc, int(chunk_size * hop_ratio), n_fft, n_mels)

                        # Find the nearest neighbor for the chunk
                        nearest_neighbor = find_nearest_neighbor(feature_vector, kd_tree, file_paths, offsets)

                        if nearest_neighbor is not None:
                            nearest_audio_file, offset = nearest_neighbor
                            print(f"Nearest neighbor file: {nearest_audio_file} at offset: {offset:.2f} seconds")

                            # Load the nearest audio chunk
                            nearest_audio_chunk, _ = librosa.load(nearest_audio_file, sr=sr, offset=offset, duration=chunk_size/sr, mono=True, res_type='kaiser_fast')

                            current_audio_chunk = nearest_audio_chunk

                    if current_audio_chunk is not None:
                        play_chunk = current_audio_chunk[:chunk_size]
                        current_audio_chunk = current_audio_chunk[chunk_size:]

                        # Apply the Hann window to the loaded audio chunk
                        windowed_chunk = play_chunk * window

                        # Normalize audio chunk to prevent clipping
                        windowed_chunk /= np.max(np.abs(windowed_chunk))

                        # Perform COLA
                        combined_chunk = np.zeros(chunk_size)
                        combined_chunk[:overlap_size] = previous_chunk[-overlap_size:] + windowed_chunk[:overlap_size]
                        combined_chunk[overlap_size:] = windowed_chunk[overlap_size:]

                        # Apply the envelope to the combined audio chunk
                        mixed_audio = apply_envelope(combined_chunk, envelope)

                        # Play back the mixed audio chunk
                        output_stream.write(mixed_audio.astype(np.float32).tobytes())

                        # Store the current chunk for overlap in the next iteration
                        previous_chunk = mixed_audio
                else:
                    # If gate is closed, write silence
                    output_stream.write(np.zeros(chunk_size, dtype=np.float32).tobytes())
            except IOError as e:
                print(f"Input overflow: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
