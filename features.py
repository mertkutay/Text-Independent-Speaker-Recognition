import numpy as np
from scipy import fftpack
from scipy.io import wavfile


def mfcc(file, num_ceps):
    rate, sig = wavfile.read(file)
    sig = pre_emphasis(sig, 0.97)
    frames = frame_signal(sig, rate, 0.025, 0.01)
    psd_frames = power_spectrum(frames, 512)
    energies = fbank_energies(psd_frames, rate)
    feats = fftpack.dct(energies, norm='ortho')[:, :num_ceps]
    feats = lifter(feats, 22)
    feats = normalize(feats)
    # tot_energies = (np.log(np.sum(psd_frames, axis=1)))
    # feats[:, 0] = tot_energies
    return feats


def dmfcc(file, num_ceps):
    m = mfcc(file, num_ceps)
    dm = delta(m)
    return np.concatenate((m, dm), axis=1)


def ddmfcc(file, num_ceps):
    m = mfcc(file, num_ceps)
    dm = delta(m)
    ddm = delta(dm)
    return np.concatenate((m, dm, ddm), axis=1)


def pre_emphasis(sig, coeff):
    return np.append(sig[0], sig[1:]-coeff*sig[:-1])


def frame_signal(sig, rate, frame_length, frame_step):
    length = int(frame_length * rate)
    step = int(frame_step * rate)
    num_frames = int(len(sig) / step)
    sig = np.concatenate((sig, np.zeros(length - len(sig) % step)))
    frames = np.empty((num_frames, length))
    for i in range(len(frames)):
        frames[i] = sig[i * step: i * step + length]
    return frames * np.hamming(length)


def power_spectrum(frames, nfft):
    fft_frames = np.fft.rfft(frames, nfft)
    return (np.absolute(fft_frames) ** 2) / nfft


def mel_fbanks(rate):
    mels = np.linspace(hz_mel(0), hz_mel(rate/2), 28)
    bins = ((512+1)*mel_hz(mels)/rate).astype(int)
    fbanks = np.zeros((len(bins)-2, bins[-1]-bins[0]+1))
    for i in range(len(fbanks)):
        fbanks[i][bins[i]: bins[i+1]] = [(k-bins[i])/(bins[i+1]-bins[i]) for k in range(bins[i], bins[i+1])]
        fbanks[i][bins[i+1]: bins[i+2]] = [(bins[i+2]-k)/(bins[i+2]-bins[i+1]) for k in range(bins[i+1], bins[i+2])]
    return fbanks


def fbank_energies(psd_frames, rate):
    fbanks = mel_fbanks(rate)
    energies = np.dot(psd_frames, fbanks.T)
    energies = np.where(energies == 0, np.finfo(float).eps, energies)
    return np.log(energies)


def lifter(feats, coeff):
    return (1 + coeff/2*np.sin(np.pi*np.arange(feats.shape[1])/coeff)) * feats


def normalize(feats):
    return (feats - np.mean(feats, axis=0))/np.var(feats, axis=0)


def delta(feats):
    delta_feats = np.empty_like(feats)
    padded_feats = np.pad(feats, ((2, 2), (0, 0)), mode='edge')
    for i in range(len(delta_feats)):
        delta_feats[i] = np.dot(np.arange(-2, 3), padded_feats[i:i+5])
    return normalize(delta_feats)


def hz_mel(f):
    return 1125*np.log(1+f/700)


def mel_hz(m):
    return 700*(np.exp(m/1125)-1)
