import os

import matplotlib.pyplot as plt
import numpy as np

"For confusion matrix plot"
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------------------
"""
Synthesis of the functions in :
- getclass : Get class name of a sound path directory.
- getname : Get name of a sound from its path directory.
- gen_allpath : Generate a matrix with path names and an array with all class names.
- plot_audio : Plot an audiosignal in time and frequency.
- plot_specgram : Plot a spectrogram (2D matrix).
- get_accuracy : Compute the accuracy between prediction and ground truth.
- show_confusion_matrix : Plot confusion matrix.
- plot_decision_boundary : Plot decision boundary of a classifier.
"""
# ----------------------------------------------------------------------------------


def getclass(sound, format=".ogg"):
    """Get class name of a sound path directory.
    Note: this function is only compatible with ESC-50 dataset path organization.
    """
    L = len(format)

    folders = sound.split(os.path.sep)
    if folders[-1][-L:] == format:
        return folders[-2].split("-")[1]
    else:
        return folders[-1].split("-")[1]


def getname(sound):
    """
    Get name of a sound from its path directory.
    """
    return os.path.sep.join(sound.split(os.path.sep)[-2:])


def gen_allpath(folder=r"Dataset_ESC-50"):
    """
    Create a matrix with path names of height H=50classes and W=40sounds per class
    and an array with all class names.
    """
    classpath = [f.path for f in os.scandir(folder) if f.is_dir()]
    classpath = sorted(classpath)

    allpath = [None] * len(classpath)
    classnames = [None] * len(classpath)
    for ind, val in enumerate(classpath):
        classnames[ind] = getclass(val).strip()
        sublist = [None] * len([f.path for f in os.scandir(val)])
        for i, f in enumerate(os.scandir(val)):
            sublist[i] = f.path
        allpath[ind] = sublist

    allpath = np.array(allpath)
    return classnames, allpath


def plot_audio(audio, audio_down, fs, fs_down):
    """
    Plot the temporal and spectral representations of the original audio signal and its downsampled version
    """
    M = fs // fs_down  # Downsampling factor

    L = len(audio)
    L_down = len(audio_down)

    frequencies = np.arange(-L // 2, L // 2, dtype=np.float64) * fs / L
    frequencies_down = (
        np.arange(-L_down // 2, L_down // 2, dtype=np.float64) * fs_down / L_down
    )
    spectrum = np.fft.fftshift(np.fft.fft(audio))
    spectrum_down = np.fft.fftshift(np.fft.fft(audio_down))

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_axes([0.0, 0.0, 0.42, 0.9])
    ax2 = fig.add_axes([0.54, 0.0, 0.42, 0.9])

    ax1.plot(np.arange(L) / fs, audio, "b", label="Original")
    ax1.plot(np.arange(L_down) / fs_down, audio_down, "r", label="Downsampled")
    ax1.legend()
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude [-]")
    ax1.set_title("Temporal signal")

    ax2.plot(frequencies, np.abs(spectrum), "b", label="Original")
    ax2.plot(
        frequencies_down, np.abs(spectrum_down) * M, "r", label="Downsampled", alpha=0.5
    )  # energy scaling by M
    ax2.legend()
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [-]")
    ax2.set_title("Modulus of FFT")
    plt.show()


def plot_specgram(
    specgram,
    ax,
    is_mel=False,
    title=None,
    xlabel="Time [s]",
    ylabel="Frequency [Hz]",
    cmap="jet",
    cb=True,
    tf=None,
):
    """Plot a spectrogram (2D matrix) in a chosen axis of a figure.
    Inputs:
        - specgram = spectrogram (2D array)
        - ax       = current axis in figure
        - title
        - xlabel
        - ylabel
        - cmap
        - cb       = show colorbar if True
        - tf       = final time in xaxis of specgram
    """
    if tf is None:
        tf = specgram.shape[1]

    if is_mel:
        ylabel = "Frequency [Mel]"
        im = ax.imshow(
            specgram,
            cmap=cmap,
            aspect="auto",
            extent=[0, tf, specgram.shape[0], 0],
            origin="lower",
        )
    else:
        im = ax.imshow(
            specgram,
            cmap=cmap,
            aspect="auto",
            extent=[0, tf, int(specgram.size / tf), 0],
            origin="lower",
        )
    fig = plt.gcf()
    if cb:
        cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('log scale', rotation=270)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return None


def get_accuracy(prediction, target):
    """
    Compute the accuracy between prediction and ground truth.
    """

    return np.sum(prediction == target) / len(prediction)
def get_precision( prediction , target, classname):
    """
    Compute the precision between prediction and ground truth.
    """
    TP = 0
    FP = 0
    for i in range(len(prediction)):
        if prediction[i] == classname and target[i] == classname:
            TP += 1
        elif prediction[i] == classname and target[i] != classname:
            FP += 1
    return TP / (TP + FP)

def get_recall(prediction, target, classname):
    """
    Compute the recall between prediction and ground truth.
    """
    TP = 0
    FN = 0
    for i in range(len(prediction)):
        if prediction[i] == classname and target[i] == classname:
            TP += 1
        elif prediction[i] != classname and target[i] == classname:
            FN += 1
    return TP / (TP + FN)
    
def get_f_score(prediction, target, classname):
    return 2/(1/get_recall(prediction, target, classname)+1/get_precision(prediction, target, classname))

def show_confusion_matrix(y_predict, y_true2, classnames, title=""):
    """
    From target and prediction arrays, plot confusion matrix.
    The arrays must contain ints.
    """

    confmat = confusion_matrix(
        y_true2, y_predict, labels=np.arange(np.max(y_true2) + 1)
    )
    heatmap(
        confmat.T,
        square=True,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=classnames,
        yticklabels=classnames,
    )
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title(title)
    plt.savefig("confusion_matrix.svg")
    plt.show()
    return None

def plot_decision_boundaries(X,y,model, ax=None, legend='',title='', s=20, N=40, cm='brg', edgc='k'):
    """
    Plot decision boundaries of a classifier in 2D, and display true labels.
    """
    if ax is None:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_axes([0.0,0.0,0.9,1.0]) 
    ax.set_aspect('equal', adjustable='box')
    # Plot the decision boundary. 
    n = 80
    vec = np.linspace(np.min(X),np.max(X),n)
    Xtmp = np.meshgrid(vec, vec)
    Xtmp2 = np.array(Xtmp).reshape(2,n**2).T
    ax.contourf(Xtmp[0], Xtmp[1], model.predict(Xtmp2).reshape(n,n), cmap=cm, alpha=0.5)
    scatterd = ax.scatter(X[:,0],X[:,1], c=y, cmap=cm, edgecolors=edgc, s=s)
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    handles, labels = scatterd.legend_elements(prop="colors")
    ax.legend(handles, legend)
