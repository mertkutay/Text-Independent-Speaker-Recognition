import numpy as np
from scipy import fftpack
from scipy.io import wavfile
import os


def mfcc(file):
    rate, sig = wavfile.read(file)
    # sig = sig / np.amax(sig)
    sig = pre_emphasis(sig, 0.97)
    # sig = speech_segmentation(sig)
    # if len(sig) < 200:
    #     return None
    frames = frame_signal(sig, rate, 0.025, 0.01)
    psd_frames = power_spectrum(frames, 512)
    energies = np.sum(psd_frames, axis=1)
    segmented = energies > np.mean(energies) / 100
    psd_frames = psd_frames[segmented]
    energies = energies[segmented]
    # energies = np.where(energies == 0, np.finfo(float).eps, energies)
    bank_energies = fbank_energies(psd_frames, rate, 26, 512)
    feats = fftpack.dct(bank_energies, type=2, norm='ortho')[:, :13]
    # feats = dist_dct(bank_energies)
    feats = lifter(feats, 22)
    feats[:, 0] = np.log(energies)
    return feats


def dmfcc(file):
    m = mfcc(file)
    # if m is None:
    #     return None
    dm = delta(m, 2)
    feats = normalize(np.concatenate((m, dm), axis=1), 6)
    return feats


def ddmfcc(file):
    m = normalize(mfcc(file), 6)
    # if m is None:
    #     return None
    dm = delta(m, 2)
    ddm = delta(dm, 2)
    feats = np.concatenate((m, dm, ddm), axis=1)
    return feats


def pre_emphasis(sig, coeff):
    return np.append(sig[0], sig[1:] - coeff * sig[:-1])


def speech_segmentation(sig):
    u = np.absolute(sig)
    s = np.zeros_like(u)
    n = np.zeros_like(u)
    tn = np.zeros_like(u)
    seg = np.zeros_like(u)
    b_s = 0.9992
    b_n = 0.9922
    b_t = 0.999975
    t_s = 2.0
    t_n = 1.414
    t_min = 0.01
    for k in range(1, len(u)-1):
        if s[k] > u[k]:
            s[k] = u[k]
        else:
            s[k] = (1 - b_s) * u[k] + b_s * s[k-1]
        if n[k] > u[k]:
            n[k] = u[k]
        else:
            n[k] = (1 - b_n) * u[k] + b_n * n[k]
        if tn[k] > n[k]:
            tn[k] = (1 - b_t) * n[k] + b_t * tn[k]
        else:
            tn[k] = n[k]
        if s[k] > t_s * tn[k] + t_min:
            seg[k] = 1
        elif s[k] < t_n * tn[k] + t_min:
            seg[k] = 0
        else:
            seg[k] = seg[k - 1]
    return sig[seg == 1]


def frame_signal(sig, rate, frame_length, frame_step):
    length = int(frame_length * rate)
    step = int(frame_step * rate)
    num_frames = int((len(sig) - 200) / step) + 1
    sig = sig[: len(sig) - ((len(sig) - 200) % step)]
    frames = np.empty((num_frames, length))
    for i in range(num_frames):
        frames[i] = sig[i * step: i * step + length]
    return frames * np.hamming(length)


def power_spectrum(frames, nfft):
    fft_frames = np.fft.rfft(frames, nfft)
    return (np.absolute(fft_frames) ** 2) / nfft


def mel_fbanks(rate, num_banks, nfft):
    mels = np.linspace(hz_mel(0), hz_mel(rate/2), num_banks+2)
    bins = ((nfft+1)*mel_hz(mels)/rate).astype(int)
    fbanks = np.zeros((len(bins) - 2, nfft//2 + 1))
    for i in range(len(fbanks)):
        fbanks[i][bins[i]: bins[i+1]] = [(k-bins[i])/(bins[i+1]-bins[i]) for k in range(bins[i], bins[i+1])]
        fbanks[i][bins[i+1]: bins[i+2]] = [(bins[i+2]-k)/(bins[i+2]-bins[i+1]) for k in range(bins[i+1], bins[i+2])]
    return fbanks


def fbank_energies(psd_frames, rate, num_banks, nfft):
    fbanks = mel_fbanks(rate, num_banks, nfft)
    energies = np.dot(psd_frames, fbanks.T)
    energies = np.where(energies == 0, np.finfo(float).eps, energies)
    return np.log(energies)


def dist_dct(energies):
    p = int(np.size(energies, 1)/2)
    feats1 = fftpack.dct(energies[:, :p], type=2, norm='ortho')
    feats2 = fftpack.dct(energies[:, p:], type=2, norm='ortho')
    feats = np.concatenate((feats1[:, 1:p], feats2[:, 1:p]), axis=1)
    return feats


def lifter(feats, coeff):
    return (1 + coeff/2*np.sin(np.pi*np.arange(feats.shape[1])/coeff)) * feats


def normalize(feats, window):
    for i in range(len(feats)):
        if i <= window:
            low_ind = 0
            high_ind = i + window + 1
        elif i + window + 1 <= len(feats):
            low_ind = i - window
            high_ind = len(feats)
        else:
            low_ind = i - window
            high_ind = i + window + 1
        feats[i] = (feats[i] - np.mean(feats[low_ind: high_ind], axis=0))/np.var(feats[low_ind: high_ind], axis=0)
    return feats


def delta(feats, window):
    denominator = 2 * sum([i ** 2 for i in range(1, window + 1)])
    delta_feats = np.empty_like(feats)
    padded_feats = np.pad(feats, ((window, window), (0, 0)), mode='edge')
    for i in range(len(delta_feats)):
        delta_feats[i] = np.dot(np.arange(-window, window + 1), padded_feats[i: i + 2 * window + 1]) / denominator
    return delta_feats


def hz_mel(f):
    return 2595*np.log10(1+f/700)


def mel_hz(m):
    return 700*(10**(m/2595)-1)
