import librosa
import numpy as np
import os
import pyaudio
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
from operator import itemgetter, attrgetter
import pickle as pk
import soundfile as sf
import sounddevice as sd
import queue
import keyboard
import time
import threading
import tkinter
import wave

TITLE = "Word Reconigtion"
RESOLUTION = "600x300"
BUTTON_CONFIG = {
    'height': 1,
    'width': 15
}
LABEL_CONFIG = {
    'wraplength': 500
}



def get_mfcc(file_path,test):
    y, sr = librosa.load(file_path) # read .wav file
    if (test):
        y, index = librosa.effects.trim(y,top_db=50,frame_length=551,hop_length=220)
        print(librosa.get_duration(y))
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix

def get_class_data(data_dir,test):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f),test) for f in files if f.endswith(".wav")]
    return mfcc

# def clustering(X, n_clusters=15):
#     kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
#     kmeans.fit(X)
#     print("centers", kmeans.cluster_centers_.shape)
#     return kmeans

# if __name__ == "__main__":
#     train_names = ["Nguoi","Duoc","Cothe","Trong","Dang"]
#     train_dataset = {}
#     for cname in train_names:
#         print(f"Load {cname} dataset")
#         train_dataset[cname] = get_class_data(os.path.join("Data", cname),False)
    
#     # Get all vectors in the train_dataset
#     all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in train_dataset.items()], axis=0)
#     print("vectors", all_vectors.shape)
#     # Run K-Means algorithm to get clusters
#     kmeans = clustering(all_vectors)
#     print("centers", kmeans.cluster_centers_.shape)
    
#     for cname in train_dataset:
#         train_dataset[cname] = list([kmeans.predict(v).reshape(-1, 1) for v in train_dataset[cname]])

#load models
train_names = ["Nguoi","Duoc","Cothe","Trong","Dang"]
models = {}
for cname in train_names:
     models[cname] = pk.load(open("Models/" + cname + ".pkl", 'rb'))
#load kmeans
kmeans = pk.load(open("Models/kmeans.pkl", 'rb'))

class Recorder:
    def __init__(self):
        self.start_button = tkinter.Button(
            root,
            text="Start Recording",
            command=self.start_recording,
            **BUTTON_CONFIG
        )
        self.start_button.pack()
        self.start_lock = False

        self.stop_button = tkinter.Button(
            root,
            text="Stop Recording",
            command=self.stop_recording,
            **BUTTON_CONFIG
        )
        self.stop_button.pack()
        self.stop_lock = True

        self.recognize_button = tkinter.Button(
            root,
            text="Recognize Word",
            command=self.recognize,
            **BUTTON_CONFIG
        )
        self.recognize_button.pack()

        self.status = tkinter.Label(
            root,
            text="No recording"
        )
        self.status.pack()
        self.recognize_lock = True

        self.is_recording = False


    def start_recording(self):
        if self.start_lock:
            return

        self.start_lock = True

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,
            frames_per_buffer=1024,
            input=True
        )

        self.frames = []

        self.is_recording = True
        self.status.config(text="Đang ghi âm")

        self.recognize_lock = True
        self.stop_lock = False

        thread = threading.Thread(target=self.record)
        thread.start()
    
    def stop_recording(self):
        if self.stop_lock:
            return

        self.stop_lock = True

        self.is_recording = False

        wave_file = wave.open("MicTest/test_mic/test.wav", "wb")

        wave_file.setnchannels(1)
        wave_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(22050)

        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        self.status.config(text="Đã ghi xong")

        self.recognize_lock = False
        self.start_lock = False
  

    def record(self):
        while (self.is_recording):
            data = self.stream.read(1024)
            self.frames.append(data)
 

    def recognize(self):
        mic_name1s = ["test_mic"]
        mic_dataset1 = {}
        for cname in mic_name1s:
            print(f"Load {cname} dataset")
            mic_dataset1[cname] = get_class_data(os.path.join("MicTest", cname),True)
            mic_dataset1[cname] = list([kmeans.predict(v).reshape(-1, 1) for v in mic_dataset1[cname]])
        
        print("Testing file mic")
        true_cname = "test_mic"
        print(true_cname)
        scores = {}
        for cname, model in models.items():
            score = model.score(mic_dataset1[true_cname][0], [len(mic_dataset1[true_cname][0])])
            scores[cname] = score

        srt = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(srt)
        self.status.config(text=f"Đây là \"{srt[0][0]}\"")

root = tkinter.Tk()
root.title(TITLE)
root.geometry(RESOLUTION)
app = Recorder()
root.mainloop()