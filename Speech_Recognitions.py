import librosa
import numpy as np
import os
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

TITLE = "Word Reconigtion"
RESOLUTION = "600x300"
BUTTON_CONFIG = {
    'height': 1,
    'width': 15
}
LABEL_CONFIG = {
    'wraplength': 500
}


def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
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

def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav")]
    return mfcc

def clustering(X, n_clusters=15):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    print("centers", kmeans.cluster_centers_.shape)
    return kmeans

if __name__ == "__main__":
    train_names = ["Nguoi","Duoc","Cothe","Trong","Dang"]
    test_names = ["test_Cothe","test_Nguoi","test_Duoc","test_Dang","test_Trong"]
    train_dataset = {}
    test_dataset = {}
    for cname in train_names:
        print(f"Load {cname} dataset")
        train_dataset[cname] = get_class_data(os.path.join("Data", cname))
        
    for cname in test_names:
        print(f"Load {cname} dataset")
        test_dataset[cname] = get_class_data(os.path.join("Data", cname))

    # Get all vectors in the train_dataset
    all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in train_dataset.items()], axis=0)
    print("vectors", all_vectors.shape)
    # Run K-Means algorithm to get clusters
    kmeans = clustering(all_vectors)
    print("centers", kmeans.cluster_centers_.shape)
    
    for cname in train_dataset:
        train_dataset[cname] = list([kmeans.predict(v).reshape(-1, 1) for v in train_dataset[cname]])
    for cname in test_dataset:
        test_dataset[cname] = list([kmeans.predict(v).reshape(-1, 1) for v in test_dataset[cname]])
    
models = {}
cname = 'Nguoi'
hmm = hmmlearn.hmm.MultinomialHMM(n_components=12, random_state=0, n_iter=1000, verbose=True, init_params='e', params='ste')
hmm.startprob_ = np.array([0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
hmm.transmat_ =np.array([
    [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ],])
if cname[:4] != 'test':
    X = np.concatenate(train_dataset[cname])
    lengths = list([len(x) for x in train_dataset[cname]])
    print("training class", cname)
    print(X.shape, lengths, len(lengths))
    hmm.fit(X, lengths=lengths)
    models[cname] = hmm
print("Training done")

cname = 'Duoc'
hmm = hmmlearn.hmm.MultinomialHMM(n_components=12, random_state=0, n_iter=1000, verbose=True, init_params='e', params='ste')
hmm.startprob_ = np.array([0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
hmm.transmat_ =np.array([
    [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ],])
if cname[:4] != 'test':
    X = np.concatenate(train_dataset[cname])
    lengths = list([len(x) for x in train_dataset[cname]])
    print("training class", cname)
    print(X.shape, lengths, len(lengths))
    hmm.fit(X, lengths=lengths)
    models[cname] = hmm
print("Training done")

cname = 'Cothe'
hmm = hmmlearn.hmm.MultinomialHMM(n_components=12, random_state=0, n_iter=1000, verbose=True, init_params='e', params='ste')
hmm.startprob_ = np.array([0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
hmm.transmat_ =np.array([
    [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ],])
if cname[:4] != 'test':
    X = np.concatenate(train_dataset[cname])
    lengths = list([len(x) for x in train_dataset[cname]])
    print("training class", cname)
    print(X.shape, lengths, len(lengths))
    hmm.fit(X, lengths=lengths)
    models[cname] = hmm
print("Training done")

cname = 'Trong'
hmm = hmmlearn.hmm.MultinomialHMM(n_components=9, random_state=0, n_iter=1000, verbose=True, init_params='e', params='ste')
hmm.startprob_ = np.array([0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
hmm.transmat_ =np.array([
    [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ],])
if cname[:4] != 'test':
    X = np.concatenate(train_dataset[cname])
    lengths = list([len(x) for x in train_dataset[cname]])
    print("training class", cname)
    print(X.shape, lengths, len(lengths))
    hmm.fit(X, lengths=lengths)
    models[cname] = hmm
print("Training done")

cname = 'Dang'
hmm = hmmlearn.hmm.MultinomialHMM(n_components=9, random_state=0, n_iter=1000, verbose=True, init_params='e', params='ste')
hmm.startprob_ = np.array([0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
hmm.transmat_ =np.array([
    [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ],])
if cname[:4] != 'test':
    X = np.concatenate(train_dataset[cname])
    lengths = list([len(x) for x in train_dataset[cname]])
    print("training class", cname)
    print(X.shape, lengths, len(lengths))
    hmm.fit(X, lengths=lengths)
    models[cname] = hmm
print("Training done")

misc = sd.query_devices()

sd.default.device = 1

q = queue.Queue()
def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

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

        self.status = tkinter.Label(
            root,
            text="No recording"
        )
        self.status.pack()

        self.recognize_button = tkinter.Button(
            root,
            text="Recognize Word",
            command=self.recognize,
            **BUTTON_CONFIG
        )
        self.recognize_button.pack()
        self.recognize_lock = True

        self.is_recording = False


    def start_recording(self):
        if self.start_lock:
            return

        self.start_lock = True

        self.is_recording = True
        self.status.config(text="Recording")

        self.recognize_lock = True
        self.stop_lock = False

        thread = threading.Thread(target=self.record)
        thread.start()

    def stop_recording(self):
        if self.stop_lock:
            return

        self.stop_lock = True

        self.is_recording = False

        self.status.config(text="Recorded")

        self.recognize_lock = False
        self.start_lock = False

    def record(self):
        file_name ="MicTest/test_mic/test.wav"
        try:
            os.remove(file_name)
        except:
            pass
        with sf.SoundFile(file_name, mode='x', samplerate=22000,
                  channels=1) as file:
            with sd.InputStream(samplerate=22000, device=sd.default.device,
                channels=1, callback=callback):
                while (self.is_recording):
                    file.write(q.get())
        print('done recording')

    def recognize(self):
        mic_name1s = ["test_mic"]
        mic_dataset1 = {}
        for cname in mic_name1s:
            print(f"Load {cname} dataset")
            mic_dataset1[cname] = get_class_data(os.path.join("MicTest", cname))
        
        for cname in mic_name1s:
            mic_dataset1[cname] = list([kmeans.predict(v).reshape(-1, 1) for v in mic_dataset1[cname]])
        
        print("Testing file mic")
        true_cname = "test_mic"
        #for true_cname in mic_name1s:
        print(true_cname)
        scores = {}
        for cname, model in models.items():
            score = model.score(mic_dataset1[true_cname][0], [len(mic_dataset1[true_cname][0])])
            scores[cname] = score

        srt = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(srt)
        self.status.config(text=f"This is \"{srt[0][0]}\"")

root = tkinter.Tk()
root.title(TITLE)
root.geometry(RESOLUTION)
app = Recorder()
root.mainloop()