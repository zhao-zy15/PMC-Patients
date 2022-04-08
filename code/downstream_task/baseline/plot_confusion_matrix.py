import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


def plot_confusion_matrix(cm, classes, file_name, normalize=True,  cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : confusion matrix
    - classes : column names
    - normalize : True: Percentage, False: count
    """
    plt.rcParams['figure.figsize'] = (3.8,2.7)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 12)
    plt.yticks(tick_marks, classes, fontsize = 12)
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=10)
    plt.ylabel('Human Annotations', fontsize = 12)
    plt.xlabel('Automated Labels', fontsize = 12)
    plt.tight_layout(pad = 0.01)
    plt.savefig(file_name, format = "pdf")
    plt.clf()

def plot_confusion_matrix_double(cm1, cm2, classes, file_name, normalize=True,  cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : confusion matrix
    - classes : column names
    - normalize : True: Percentage, False: count
    """
    plt.rcParams['figure.figsize'] = (5,3)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    if normalize:
        cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
        cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm1)
    print(cm2)
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.imshow(cm1, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(classes, fontsize = 12)
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(classes, fontsize = 12)
    fmt = '.3f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        ax1.text(j, i, format(cm1[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm1[i, j] > thresh else "black",
                 fontsize=12)
    ax1.set_ylabel('Human annotations', fontsize = 14)
    ax1.set_xlabel('PMC-P relevance', fontsize = 14)

    cm = ax2.imshow(cm2, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(classes, fontsize = 12)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(classes, fontsize = 12)
    fmt = '.3f' if normalize else 'd'
    thresh = cm2.max() / 2.
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        ax2.text(j, i, format(cm2[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm2[i, j] > thresh else "black",
                 fontsize=12)
    ax2.set_xlabel('PMC-P similarity', fontsize = 14)
    plt.savefig(file_name, format = "pdf")
    plt.clf()


if __name__ == '__main__':
    #cm = np.array([[9012, 467, 0], [1226, 6315, 123], [4, 131, 1680]])
    #plot_confusion_matrix(cm, [0, 1, 2], "../../../figures/Confusion_matrix_PPS.pdf", True)

    #cm = np.array([[597, 4, 5], [2, 1494, 60], [15, 126, 4800]])
    #plot_confusion_matrix(cm, ["B", "I", "O"], "../../../figures/Confusion_matrix_PNR.pdf", True)

    #cm = np.array([[980, 2], [1709, 339]])
    #plot_confusion_matrix(cm, ["0", "1"], "../../../figures/Confusion_matrix_PAR_human.pdf", True)

    cm2 = np.array([[814, 4], [1811, 401]])
    cm1 = np.array([[980, 2], [1709, 339]])
    plot_confusion_matrix_double(cm1, cm2, [0, 1], "../../../figures/Confusion_matrix_PPR_PAR.pdf", True)