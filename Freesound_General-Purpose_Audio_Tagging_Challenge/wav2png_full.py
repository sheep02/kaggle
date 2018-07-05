from os import path
from io import BytesIO
import numba

import numpy as np
import pandas as pd

import librosa
from librosa import display

import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import cv2


def max_var_interval(data, len_sum=1000):
    average = np.mean(data)
    max_var = 0
    max_var_idx = 0

    for startidx in range(0, int(len(data)-len_sum), int(len_sum/4)):
        var = np.mean([(d-average)**2 for d in data[startidx:startidx+len_sum]])

        if max_var < var:
            max_var = var
            max_var_idx = startidx

    return max_var_idx


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)

    return cv2.imdecode(img_array, cv2_img_flag)


def save_grapgh(dist_path, fname, x1=0, x2=None, y1=0, y2=None, size_resize=(64, 64), tag=""):
    imgdata = BytesIO()
    plt.savefig(imgdata, format='png')
    img = create_opencv_image_from_stringio(img_stream=imgdata, cv2_img_flag=cv2.IMREAD_COLOR)

    if None == x2 or None == y2:
        cv2.imwrite(path.join(dist_path, fname.replace(".wav", f"{tag}.png")), cv2.resize(img[y1:, x1:], size_resize))
    else:
        cv2.imwrite(path.join(dist_path, fname.replace(".wav", f"{tag}.png")), cv2.resize(img[y1:y2, x1:x2], size_resize))


def wav2img(dist_path, src_path, fnames, mode=None, size_resize=(64, 64), audio_length=32000, resamplingrate=16000, hop_length=512):
    if not mode in ["CQT", "TEMPO", "HPSS"]:
        return

    for fname in tqdm(fnames):
        if fname in ["0b0427e2.wav", "6ea0099f.wav", "b39975f5.wav"]:
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype='uint8'))
            img.save(path.join(dist_path, fname.replace(".wav", ".png")))

            continue

        raw, _ = librosa.load(path.join(src_path, fname), sr=resamplingrate)

        if "CQT" == mode:
            C = librosa.cqt(raw, sr=resamplingrate)

            plt.clf() 
            display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=resamplingrate, x_axis='time', y_axis='cqt_note')

            save_grapgh(dist_path, 81, 577, 58, 428)

        if "TEMPO" == mode:
            oenv = librosa.onset.onset_strength(
                y=raw, 
                sr=resamplingrate, 
                hop_length=hop_length
            )
            
            tempogram = librosa.feature.tempogram(
                onset_envelope=oenv, 
                sr=resamplingrate, 
                hop_length=hop_length
            )

            plt.clf()            
            display.specshow(
                tempogram, 
                sr=resamplingrate, 
                hop_length=hop_length, 
            )

            save_grapgh(dist_path, 79, 498, 57, 371)

        if "HPSS" == mode:
            D = librosa.stft(raw)
            H, P = librosa.decompose.hpss(D)

            plt.clf()
            display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=resamplingrate)
            save_grapgh(dist_path, fname, 79, 498, 57, 371)
            
            plt.clf()
            display.specshow(librosa.amplitude_to_db(P, ref=np.max))
            save_grapgh(dist_path, fname, 79, 498, 57, 371, tag="_p")
            
            plt.clf()
            display.specshow(librosa.amplitude_to_db(H, ref=np.max))
            save_grapgh(dist_path, fname, 79, 498, 57, 371, tag="_h")


                
path_stft = r"G:\kaggle\sound\img_stft\train64"
path_hpss = r"G:\kaggle\sound\img_hpss\train64"
path_tempo = r"G:\kaggle\sound\img_tempo\train64"

path_stft_test = r"G:\kaggle\sound\img_stft\test64"
path_hpss_test = r"G:\kaggle\sound\img_hpss\test64"
path_tempo_test = r"G:\kaggle\sound\img_tempo\test64"


test = pd.read_csv("G:/kaggle/sound/_dl/sample_submission.csv")
train = pd.read_csv('G:/kaggle/sound/_dl/train.csv')


wav2img(
    path_hpss, 
    "G:/kaggle/sound/_dl/audio_train/", 
    train.fname.values, 
    mode="HPSS",
    size_resize=(64, 64),
    audio_length=32000, 
    resamplingrate=16000
)


"""

r"G:\kaggle\sound\tempo_img\train64"
"G:/kaggle/sound/cqt_img/png_test64_full/"

"""