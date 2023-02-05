# in the name of god

# import library 

from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
import math

def generate_final_signal(sound, fs):

    # load sound
    samplerate, signal = wavfile.read(sound)

    # seperate L/R sound and merge 
    left_signal = signal[:,0]
    right_signal = signal[:,1]

    merged_signal = (left_signal + right_signal) / 2

    # sampling from analog signal

    new_samplerate = int(len(merged_signal)/fs)
    final_signal = np.zeros(new_samplerate, dtype=int)

    for i in range(len(merged_signal)):
        if i % fs == 0:
            final_signal[i//fs] = merged_signal[i]
    return final_signal, samplerate


def quantization(final_signal, v):
    x_max = max(final_signal)
    x_min = min(final_signal)

    delta = x_max / 2**(v-1)
    
    treshold = []
    for i in range(2**v + 1):
        treshold.append(x_min + i*delta)
    
    quantized_signal = np.zeros(len(final_signal), dtype=float)
    
    for j in range(len(final_signal)):
        for i in range(2**v):
            if int(treshold[i])  <= final_signal[j] < int(treshold[i+1]):
                quantized_signal[j] = i

    return quantized_signal

def nbc_generator(v):
    nbc = []
    for n in range(2**v):
        nbc.append(bin(n).replace("0b", "0"*(v-len(bin(n))+2)))
    return nbc 

def gray_generator(v):
    gray = []
    gray.append("0")
    gray.append("1")
    i = 2
    j = 0 
    while (True):
        if i >= 1 << v:
            break
        for j in range(i - 1, -1, -1):
            gray.append(gray[j])
        
        for j in range(i, 2 * i):
            gray[j] = "1" + gray[j]
        i = i << 1
    for i in gray:
        gray[i] = '0'*(v - len(gray[i])) + gray[i]
    return gray

def generate_bit_stream(quantized_signal, model,v):
    if model == 'NBC':
        refrence = nbc_generator(v)
    
    if model == 'GRAY':
        refrence = gray_generator(v)
    
    list_bit_stream = []
    for i in range(len(quantized_signal)):
        list_bit_stream.append(refrence[int(quantized_signal[i])])
    
    bit_stream = ''
    for bit in list_bit_stream:
        bit_stream = bit_stream + bit
    return bit_stream , refrence
    
v = 5
fs = 4
def ADC(sound, fs, v, model):
    final_signal, samplerate = generate_final_signal(sound, fs)
    quantized_signal = quantization(final_signal, v)
    bit_stream, refrence = generate_bit_stream(quantized_signal,model,v)
    return final_signal, quantized_signal, bit_stream, refrence, samplerate

final_signal, quantized_signal, bit_stream, refrence, samplerate = ADC('sound.wav', fs, v, 'NBC')
# print(max(final_signal), max(quantized_signal), bit_stream)
max_signal = max(final_signal)

def reverse_bit_stream(bit_stream, refrence, v):
    tmp = []
    for i in range(len(bit_stream)//v):
       tmp.append(bit_stream[i*v:(i+1)*v])
    for i in range(len(tmp)):
        for j in range(len(refrence)):
            if refrence[j] == tmp[i]:
                tmp[i] = j
    reverse_quantized_signal = np.array(tmp) - (2**(v-1))
    return reverse_quantized_signal
# reverse_quantized_signal = reverse_bit_stream(bit_stream, refrence, v)

def generate_reverse_signal(reverse_quantized_signal, max_signal):
    reverse_signal = reverse_quantized_signal * max_signal / 2**(v-1)
    return reverse_signal


# def low_pass_filter(signal, cutoff, fs, oreder=5):
#     fourier_signal = np.fft.fft(signal)
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(oreder, normal_cutoff, btype='low', analog=False)
#     filtered_fourier_signal = lfilter(b, a, fourier_signal)
#     reverse_signal_final = np.fft.ifft(filtered_fourier_signal)
#     return reverse_signal_final

cutoff = 1

def DAC(bit_stream, refrence, v, max_signal, output_sound, samplerate):
    reverse_quantized_signal = reverse_bit_stream(bit_stream, refrence, v)
    reverse_signal = generate_reverse_signal(reverse_quantized_signal, max_signal)
    # reverse_signal_final = low_pass_filter(reverse_signal, cutoff, fs, oreder=5)
    wavfile.write(output_sound, samplerate, abs(reverse_signal))
    return reverse_quantized_signal, reverse_signal

reverse_quantized_signal, reverse_signal=  DAC(bit_stream, refrence, v, max_signal, 'output_sound.wav', max_signal)
print(max(reverse_quantized_signal))