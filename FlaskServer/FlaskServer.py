import socket
import struct
import traceback
import logging
import time
import numpy as np
from flask import Flask, jsonify
import pickle
import librosa ###Mover La libreria a la carpeta del ejecutable
import parselmouth
import scipy ###Sustituir la carpeta del ejecutable por la original de la carpeta venv\lib
import speech_recognition as spr
from numpy import asarray
import cv2 ###Sustituir la carpeta del ejecutable por la original de la carpeta venv\lib
###Sustituir la carpeta scikitlearn del ejecutable por la original de la carpeta venv\lib
from deepface import DeepFace
import text2emotion as te
from googletrans import Translator
import os
import sys
'''
s = socket.socket()
socket.setdefaulttimeout(None)
print('socket created ')
_port = 60000
s.connect(('127.0.0.1', _port))  # local host

request_code = np.array([1001], np.float64)

s.sendall(request_code)  # sending back
bytes_received = s.recv(40000)  # received bytes
response = bytes_received.decode('UTF-8')  # converting into string
print(response)

file_dir = response
'''

##import nltk
##nltk.download()

file_dir = os.path.join(os.getenv('APPDATA'),'AudioRecognition')

audio_path = os.path.join(file_dir, 'RecordedAudio.wav')
model_path = os.path.join(file_dir, 'SVM_Model_New.pkl')

img_path = os.path.join(file_dir, 'picture')
number_Path = os.path.join(file_dir, 'number.txt')
n = open(number_Path)
number = int(n.read())

LanguajeTxt = open(os.path.join(file_dir, 'Languaje.txt'))
Languaje = LanguajeTxt.read()

app = Flask(__name__)
model = pickle.load(open(model_path, 'rb'))
r = spr.Recognizer()
#s.close()

@app.route('/')
def predict():
    y, sr = librosa.load(audio_path)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)
    desEst = np.std(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    kurtosis = scipy.stats.kurtosis(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    skew = scipy.stats.skew(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    mfccFeatures = np.concatenate([mfcc, delta, delta_delta, desEst, kurtosis, skew])

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean = np.mean(cent, axis=1)
    stDev = np.std(cent, axis=1)
    kurtosis = scipy.stats.kurtosis(cent, axis=1)
    skew = scipy.stats.skew(cent, axis=1)
    maxValue = np.amax(cent)
    minValue = np.amin(cent)

    join = np.concatenate([mean, stDev, kurtosis, skew])
    centroidFeatures = np.append(join, [maxValue, minValue])

    flatness = librosa.feature.spectral_flatness(y=y)
    mean = np.mean(flatness, axis=1)
    stDev = np.std(flatness, axis=1)
    kurtosis = scipy.stats.kurtosis(flatness, axis=1)
    skew = scipy.stats.skew(flatness, axis=1)
    maxValue = np.amax(flatness)
    minValue = np.amin(flatness)

    join = np.concatenate([mean, stDev, kurtosis, skew])
    flatnessFeatures = np.append(join, [maxValue, minValue])

    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mean = np.mean(contrast, axis=1)
    stDev = np.std(contrast, axis=1)
    kurtosis = scipy.stats.kurtosis(contrast, axis=1)
    skew = scipy.stats.skew(contrast, axis=1)
    maxValue = np.amax(contrast)
    minValue = np.amin(contrast)

    join = np.concatenate([mean, stDev, kurtosis, skew])
    contrastFeatures = np.append(join, [maxValue, minValue])

    lpc = librosa.lpc(y, 16)
    tempo = librosa.beat.tempo(y=y, sr=sr)

    fichero_Audio = parselmouth.Sound(audio_path)
    Extract_intensity = fichero_Audio.to_intensity()
    Mean_intensity = np.mean(Extract_intensity.values.T)
    intensity = np.array([Mean_intensity])

    Extract_pitch = fichero_Audio.to_pitch()
    pitch_values = Extract_pitch.selected_array['frequency']
    contador = 1
    suma = 0
    for pitch_value in pitch_values:
        if pitch_value > 0:
            suma += pitch_value
            contador += 1

    media = suma / contador
    pitch = np.array([media])

    join = np.concatenate(
        [mfccFeatures, centroidFeatures, flatnessFeatures, contrastFeatures, lpc, intensity, pitch, tempo])
    data = asarray([join])
    print(data.shape)
    prediction = model.predict(data)

    with spr.AudioFile(audio_path) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        try:
            text = r.recognize_google(audio_data, language=Languaje)
        except:
            noText = True
            text = 'no text'
            return 'NoText'
        translator = Translator()
        traduccion = translator.translate(text).text
        emocion = te.get_emotion(traduccion)
        if emocion['Angry'] >= emocion['Fear'] and emocion['Angry'] >= emocion['Happy'] and emocion['Angry'] >= emocion['Sad'] and emocion['Angry'] >= emocion['Surprise']:
            TranscriptionEmotion = 1
        else:
            TranscriptionEmotion = 0

    angryCount = 0
    nullCount = 0
    photoemotions = ""
    for n in range(number):
        path = img_path + str(n) + '.jpg'
        img = cv2.imread(path)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        try:
            predictions = DeepFace.analyze(img, actions=['emotion'])
        except:
            predictions = {'dominant_emotion': 'null'}
        photoemotions += ' Photo ' + str(n + 1) + ': ' + str(predictions['dominant_emotion']) + ','
        # print(predictions['dominant_emotion'])
        if predictions['dominant_emotion'] == 'angry':
            angryCount += 1
        if predictions['dominant_emotion'] == 'null':
            nullCount += 1

    if (angryCount >= number / 2):
        FaceEmotion = 1
    else:
        FaceEmotion = 0
    if(nullCount >= number/2):
        return 'NoFace'
    output = 'Audio emotion: ' + str(prediction[0]) + '; Transcription: ' + text + '; Traduccion: '+ traduccion + '; Transcription emotion: ' + str(emocion) + photoemotions

    print(output)
    ##return jsonify(output)
    result = '' + str(prediction[0]) + ';' + str(TranscriptionEmotion) + ';' + str(FaceEmotion)
    return result
if __name__ == '__main__':
    app.run(port=5000, debug=True)

print('Finished')