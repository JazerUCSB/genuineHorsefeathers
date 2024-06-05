import os
import numpy as np
import pyaudio
import librosa
from sklearn.neighbors import KDTree
import joblib
import time
import soundfile as sf
from soundfile import SoundFile

# Define constants
chunk_size = 1024  # Adjusted buffer size
hop_length = chunk_size // 2
n_mfcc = 13
n_fft = 1024  # Adjust this to be smaller for better time resolution and pitch capture
n_mels = 40  # Reduce the number of Mel filters to reduce computational load
sr = 44100
overlap_size = chunk_size // 2

# Define noise gate thresholds
threshold_on = 0.00
threshold_off = 0.00
smoothing_factor = 0.01

window = np.hanning(chunk_size)

def extract_features(y, sr, n_mfcc, hop_length, n_fft, n_mels):
    # Compute the MFCC features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=chunk_size, n_fft=n_fft, n_mels=n_mels), axis=1)

    # Calculate the fundamental frequency (F0)
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=chunk_size))
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
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=chunk_size, n_fft=n_fft), axis=1)
    
    # Compute Spectral Roll-off
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=chunk_size, n_fft=n_fft), axis=1)

    # Combine MFCC, F0, Spectral Centroid, and Spectral Roll-off features
    features = np.concatenate([mfcc, [np.mean(f0)], spectral_centroid, spectral_rolloff])

    return features

def build_kd_tree(corpus_folder, sr):
    features = []
    big_wave = []

    for root, _, files in os.walk(corpus_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                with SoundFile(file_path) as f:
                    for block in f.blocks(blocksize=chunk_size, fill_value=0):
                        big_wave.append(block)

    for chunk in big_wave:
        chunk_windowed = chunk * window
        feature_vector = extract_features(chunk_windowed, sr, n_mfcc, hop_length, n_fft, n_mels)
        features.append(feature_vector)
    
    kd_tree = KDTree(np.array(features))
    return kd_tree, big_wave

def build_big_wave(corpus_folder):
    big_wave = []
    for root, _, files in os.walk(corpus_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                with SoundFile(file_path) as f:
                    for block in f.blocks(blocksize=chunk_size, fill_value=0):
                        big_wave.append(block)
    return big_wave

def save_kd_tree(corpus_folder, kd_tree):
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')
   
    
    joblib.dump(kd_tree, kd_tree_file)
    

def load_kd_tree(corpus_folder):
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')
    
    
    kd_tree = joblib.load(kd_tree_file)
    
    
    return kd_tree

def find_nearest_neighbor(feature_vector, kd_tree, big_wave):
    try:
        dist, ind = kd_tree.query([feature_vector], k=1) 
        return big_wave[ind[0][0]]
    except Exception as e:
        print(f"Error finding nearest neighbor: {e}")
        return None

def apply_envelope(audio, envelope):
    return audio * envelope



def main():
    corpus_folder = input("Please provide the path to the folder containing WAV files: ")
    
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')

    if os.path.exists(kd_tree_file):
        print("Loading KD-tree from files...")
        kd_tree = load_kd_tree(corpus_folder)
        big_wave = build_big_wave(corpus_folder)
        print("KD-tree loaded successfully.")
    else:
        print("Building KD-tree...")
        kd_tree, big_wave = build_kd_tree(corpus_folder, sr)
        save_kd_tree(corpus_folder, kd_tree)
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
    
    previous_rms = 0.0

    try:
        previous_chunk = np.zeros(chunk_size, dtype=np.float32)
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
                        feature_vector = extract_features(audio_chunk_windowed, sr, n_mfcc, hop_length, n_fft, n_mels)

                        # Find the nearest neighbor for the chunk
                        nearest_neighbor = find_nearest_neighbor(feature_vector, kd_tree, big_wave)


                    if nearest_neighbor is not None:
                        
                        # Apply the Hann window to the loaded audio chunk
                        windowed_chunk = nearest_neighbor * window

                        # Normalize audio chunk to prevent clipping
                        windowed_chunk /= .002 + np.max(np.abs(windowed_chunk))

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
