import librosa
import numpy
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

dataset_folder = './data'
souce_data_folder = './source'

OldMax = 20
OldMin = -92
NewMax = 1
NewMin = 0


def change_range(old):
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    output = ((old - OldMin) * NewRange) / OldRange + NewMin

    return output


def add_zerous(log_mel_spectrogram):
    zerous_count = 33 - log_mel_spectrogram.shape[1]
    left_zerous = zerous_count // 2
    log_mel_spectrogram = numpy.pad(log_mel_spectrogram, [(
        0, 0), (left_zerous, zerous_count - left_zerous)], mode='constant')
    return log_mel_spectrogram


def save_data(source_path, mode='train'):
    cur_file = f'{mode}.txt'
    dest_path = f'{dataset_folder}/{mode}'
    onlyfiles = [f for f in listdir(
        source_path) if isfile(join(source_path, f))]
    descpfile = open(f'{dest_path}/{cur_file}', 'a')

    for file in onlyfiles:
        filename = file[:-4]
        audio_file = f'{source_path}/{file}'
        audio, sample_rate = librosa.load(audio_file)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=90)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        log_mel_spectrogram = change_range(log_mel_spectrogram)

        if log_mel_spectrogram.size < 90 * 33:
            log_mel_spectrogram = add_zerous(log_mel_spectrogram)

        if log_mel_spectrogram.size > 90 * 33:
            continue

        descpfile.write(f'{filename}\n')
        numpy.save(f'{dest_path}/elements/{filename}', log_mel_spectrogram)
        f = open(f"{dest_path}/text/{filename}.txt", 'w')
        f.write(filename[0])
        f.close()
    descpfile.close()


def convert(filepath):
    audio_file = filepath
    audio, sample_rate = librosa.load(audio_file)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=90)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    log_mel_spectrogram = change_range(log_mel_spectrogram)

    if log_mel_spectrogram.size <= 90 * 33:
        log_mel_spectrogram = add_zerous(log_mel_spectrogram)
    else:
        return None

    return log_mel_spectrogram


def clean_data(mode='train', to_clean='all'):
    to_delete = '*'
    if to_clean == 'danil':
        to_delete = '*danil*'

    path = Path(f'{dataset_folder}/{mode}/elements')
    for file_name in path.glob(to_delete):
        file_name.unlink()

    path = Path(f'{dataset_folder}/{mode}/text')
    for file_name in path.glob(to_delete):
        file_name.unlink()

    newlines = []
    if to_clean == 'danil':
        file_read = open(f'{dataset_folder}/{mode}/{mode}.txt', 'r')
        lines = file_read.readlines()
        file_read.close()
        for line in lines:
            if not to_delete in line:
                newlines.append("")
    else:
        newlines = ['']

    write_file = open(f'{dataset_folder}/{mode}/{mode}.txt', 'w')
    write_file.writelines(newlines)
    write_file.close()


mode = 'train'

if __name__ == '__main__':
    clean_data()
    modes = ['train', 'test']
    
    for cur_mode in modes:
        if cur_mode == 'train':
            for i in range(1, 61):
                path = f'{souce_data_folder}/{i}'
                if i < 10:
                    path = f'{souce_data_folder}/0{i}'
                save_data(path)
        else:
            test_dirs = ['13', '24', '54', '43', '04', '45', '32']

            for dir in test_dirs:
                path = f'{souce_data_folder}/{dir}'
                save_data(path, mode='test')
