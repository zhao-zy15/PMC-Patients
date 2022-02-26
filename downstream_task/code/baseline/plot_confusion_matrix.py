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
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.tight_layout(pad = 0.01)
    plt.savefig(file_name, format = "pdf")
    plt.clf()

if __name__ == '__main__':
    cm = np.array([[9012, 467, 0], [1226, 6315, 123], [4, 131, 1680]])
    plot_confusion_matrix(cm, [0, 1, 2], "Confusion_matrix_PPS.pdf", True)

    cm = np.array([[597, 4, 5], [2, 1494, 60], [15, 126, 4800]])
    plot_confusion_matrix(cm, ["B", "I", "O"], "Confusion_matrix_PNR.pdf", True)
