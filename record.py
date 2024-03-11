import pyaudio
import wave
import audioop
import math
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import utils
import shutil
import os
from pathlib import Path


def delete(path):
    dir = Path(path)
    for file_name in dir.glob('*danil*'):
        file_name.unlink()

utils.clean()
path = './voice/'
delete(path)

for i in range(2):
    filename = f"0_danil{i}.wav"

    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = 0.750
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)

    frames = []

    print("Запись")

    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        rms = audioop.rms(data, 2)
        decibel = 20 * math.log10(rms)
        print(f'{int(decibel)} дБ')
        if decibel >= 90:
            print('Все хорошо?')
        frames.append(data)

    print('Запись завершена')

    stream.stop_stream()
    stream.close()
    wf = wave.open(path + filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

utils.save_data(path)

