#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy import stats
import scipy.io.wavfile


def analyze_wav_file(wav_file):
    s, fs = sf.read(wav_file)
    wlen = 25e-3 * fs
    wshift = 10e-3 * fs
    woverlap = wlen - wshift

    #funkcia pre spectogram doplnena o hammingovo okno, dlzku, prekrytie a posun
    #dlzka = 25ms prekrytie = woverlap [15ms]
    f, t, sgr = spectrogram(s, fs, window='hamming', nperseg=int(wlen), noverlap=woverlap, nfft=511)

    plt.clf()
    t_1 = np.arange(s.size) / fs
    plt.figure(figsize=(6, 3))
    plt.plot(t_1, s)
    plt.gca().set_ylabel('Signal')
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('"Development" and "Psychological" vs. {}'.format(wav_file))
    axes = plt.gca()
    axes.set_xlim([0, len(t) / 100])
    plt.tight_layout()
    plt.savefig('{}_analyse.jpg'.format(wav_file))
    plt.clf()

    sgr_log = 10 * np.log10(sgr + 1e-20)
    features = np.zeros((16, len(t)))
    help_for_features_spectogram = list(range(16))

    #funkcia na zredukovanie 256 -> 16 riadkov
    def fill_features_array(start, stop, which_row):
        j = start
        i = 0
        while i < len(t):
            while j < stop:
                features[which_row][i] += sgr_log[j][i]
                j = j + 1
            j = start
            i = i + 1

    #naplnenie zredukovaneho spektogramu
    row_new = 0
    row_old = 0
    while row_new < 16:
        fill_features_array(row_old, row_old+15, row_new)
        row_old += 16
        row_new += 1

    #transpose aby sme mohli do pearsona posielat stlpce tak ako potrebujeme
    transpose_for_pearson = np.transpose(features)

    #spectogram pre celu vetu
    #vacsia cast prevzata od Ing. Katerina Zmolikova
    plt.clf()
    plt.figure(figsize=(9, 3))
    plt.pcolormesh(t, f, sgr_log, cmap='autumn')
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig('{}1.jpg'.format(wav_file))
    plt.clf()

    #redukovany spectogram pre celu vetu
    plt.clf()
    plt.figure(figsize=(9, 3))
    plt.pcolormesh(t, help_for_features_spectogram, features, cmap='autumn')
    plt.gca().set_xlabel('t [s]')
    plt.gca().set_ylabel('Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('{}2.jpg'.format(wav_file))
    plt.clf()
    return features, transpose_for_pearson, t


def count_pearson_coeficients(sentence_wav, query1_wav, query2_wav):
    sentence_data, fs = sf.read(sentence_wav)
    query1_data, fs = sf.read(query1_wav)
    query2_data, fs = sf.read(query2_wav)

    feature, transposed_feature, t1 = analyze_wav_file(sentence_wav)
    query1, transposed_query1, t2 = analyze_wav_file(query1_wav)
    query2, transposed_query2, t3 = analyze_wav_file(query2_wav)

    score1 = []
    first1 = 0
    for i1 in range(len(t1)-len(t2)):
        summary1 = 0
        for j1 in range(len(t2)):
            tosummary1, useless1 = stats.pearsonr(transposed_query1[j1], transposed_feature[i1+j1])
            summary1 += tosummary1
        score1.append(summary1/transposed_query1.shape[0])
        if summary1/transposed_query1.shape[0] >= 0.90 and first1 == 0:
            scipy.io.wavfile.write('./hits/q1_{}.wav'.format(sentence_wav), fs, sentence_data[i1*10*16:i1*10*16+len(query1_data)])
            first1 += 1

    score2 = []
    first2 = 0
    for i2 in range(len(t1) - len(t3)):
        summary2 = 0
        for j2 in range(len(t3)):
            tosummary2, useless2 = stats.pearsonr(transposed_query2[j2], transposed_feature[i2+j2])
            summary2 += tosummary2
        score2.append(summary2/transposed_query2.shape[0])
        if summary2/transposed_query2.shape[0] >= 0.90 and first2 == 0:
            scipy.io.wavfile.write('./hits/q2_{}.wav'.format(sentence_wav), fs, sentence_data[i2*10*16:i2*10*16+len(query2_data)])
            first2 += 1

    return score1, score2, t1


def find_word_in_sentence(sentence_wav, query1_wav, query2_wav):
    score1, score2, t1 = count_pearson_coeficients(sentence_wav, query1_wav, query2_wav)
    plt.plot(np.arange(len(score1)) / 100, score1, label='Development')
    plt.plot(np.arange(len(score2)) / 100, score2, label='Psychological')
    plt.legend()
    plt.gca().set_ylabel('Scores')
    plt.gca().set_xlabel('t [s]')
    plt.ylim(top=1.0)
    axes = plt.gca()
    axes.set_xlim([0, len(t1)/100])
    plt.tight_layout()
    plt.savefig('{}_final.jpg'.format(sentence_wav))


###MAIN###
find_word_in_sentence('sa1.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('sa2.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('si1306.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('si1936.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('si676.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('sx136.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('sx226.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('sx316.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('sx406.wav', 'q1.wav', 'q2.wav')
find_word_in_sentence('sx46.wav', 'q1.wav', 'q2.wav')
