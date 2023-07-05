"""
SCRIPT NÚMERO: 1
Este script calcula el Espectrograma de Mel y las matriz de Lag para TODAS las 
canciones de SALAMI de acorde a los contextos de lag L = 14s y 88s tal y como
se describe en el paper "MUSIC BOUNDARY DETECTION USING NEURAL NETWORKS ON 
SPECTROGRAMS AND SELF-SIMILARITY LAG MATRICES"
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
import skimage.measure
import scipy
from scipy.spatial import distance
import os

start_time = time.time()


"=================================Functions================================="
def fourier_transform(sr_desired, name_song, window_size, hop_length):
    "This function calculates the mel spectrogram in dB with Librosa library"
    y, sr = librosa.load(name_song, sr=None)
    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired
    stft = np.abs(librosa.stft(y=y, n_fft=window_size, hop_length=hop_length))
    return stft 

def max_pooling(stft, pooling_factor):
    x_prime = skimage.measure.block_reduce(stft, (1, pooling_factor), np.max) 
    return x_prime
  
    
def sslm(stft, sr_desired, pooling_factor, lag):
    
    padding_factor = lag
    
    """"This part pads a mel spectrogram gived the spectrogram a lag parameter 
    to compare the first rows with the last ones and make the matrix circular""" 
    pad = np.full((stft.shape[0], padding_factor), -70) #matrix of 80x30frames of -70dB corresponding to padding
    S_padded = np.concatenate((pad, stft), axis=1) #padding 30 frames with noise at -70dB at the beginning

    """This part max-poolend the spectrogram in time axis by a factor of p"""
    x_prime = skimage.measure.block_reduce(S_padded, (1,p), np.max) #Mel Spectrogram downsampled

    PCPs = librosa.feature.chroma_stft(S=x_prime, sr=sr_desired, n_fft=window_size, hop_length=hop_length)
    PCPs = PCPs[1:,:]

    #Bagging frames
    m = 2 #baggin parameter in frames
    x = [np.roll(PCPs,n,axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)

    #Cosine distance calculation: D[N/p,L/p] matrix
    distances = np.zeros((x_hat.shape[1], padding_factor//p)) #D has as dimensions N/p and L/p
    for i in range(x_hat.shape[1]): #iteration in columns of x_hat
        for l in range(padding_factor//p):
            if i-(l+1) < 0:
                cosine_dist = 1
            elif i-(l+1) < padding_factor//p:
                cosine_dist = 1
            else:
                cosine_dist = distance.cosine(x_hat[:,i], x_hat[:,i-(l+1)]) #cosine distance between columns i and i-L
            distances[i,l] = cosine_dist
      
    #Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1
    epsilon = np.zeros((distances.shape[0], padding_factor//p)) #D has as dimensions N/p and L/p
    for i in range(padding_factor//p, distances.shape[0]): #iteration in columns of x_hat
        for l in range(padding_factor//p):
            epsilon[i,l] = np.quantile(np.concatenate((distances[i-l,:], distances[i,:])), kappa)

    distances = distances[padding_factor//p:,:]  
    epsilon = epsilon[padding_factor//p:,:] 
    x_prime = x_prime[:,padding_factor//p:] 

    #Self Similarity Lag Matrix
    sslm = scipy.special.expit(1-distances/epsilon) #aplicación de la sigmoide
    sslm = np.transpose(sslm)
    sslm = skimage.measure.block_reduce(sslm, (1, 3), np.max)
    
    #Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(sslm.shape[0]):
        for j in range(sslm.shape[1]):
            if np.isnan(sslm[i,j]):
                sslm[i,j] = 0

    return sslm

"=================================Variables================================="
window_size = 2048 #(samples/frame)
hop_length = 1024 #overlap 50% (samples/frame) 
sr_desired = 44100
p = 2 #pooling factor
L_sec_near = 14 #lag near context in seconds
L_near = round(L_sec_near*sr_desired/hop_length) #conversion of lag L seconds to frames


"=================================Paths====================================="
song_path = "E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\songs\\"
im_path_L_chroma_near = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from Chromas cosine 2pool3\\"

if not os.path.exists(im_path_L_chroma_near):
    os.makedirs(im_path_L_chroma_near)

 
"===============================Main Loop===================================="    
i = 0
for root, dirs, files in os.walk(song_path):
    for name_song in files:
        start_time_song = time.time()
        i += 1
        song_id = name_song[:-4] #delete .mp3 characters from the string
        print("song", song_id, "prepared to be processed.")
        if str(song_id) + ".npy" not in os.listdir(im_path_L_chroma_near):
            stft = fourier_transform(sr_desired, song_path + name_song, window_size, hop_length)
            sslm_near = sslm(stft, sr_desired, p, L_near)
            
            #Save mels matrices and sslms as numpy arrays in separate paths 
            np.save(im_path_L_chroma_near + song_id, sslm_near)
                    
            print("song", song_id, "converted.",  i,"/",len(files), "song converted to all files in : {:.2f}s".format(time.time() - start_time_song))
        
        else:
            print("song", song_id, "already in the directory.")
        
print("All images have benn converted in: {:.2f}s".format(time.time() - start_time))
