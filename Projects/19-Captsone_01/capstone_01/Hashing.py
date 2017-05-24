from PIL import Image as _PLImage
import dhash as _dhash
import numpy as _np
from keras.datasets import mnist as _mnist
from sklearn import metrics

def load__mnist():
    '''
    X_train, Y_train, X_test, Y_test = load__mnist()
    '''
    (X_train, Y_train), (X_test, Y_test) = _mnist.load_data()
    return X_train, Y_train, X_test, Y_test

def difference_hash(Images, size=8):
    '''
    Difference perceptual hashing.
    ------------------------------
    I_nput:
    Images (Int) = 3D Matrix of integers where :
                    - dim 01 = instances
                    - dim 02 = rows
                    - dim 02 = cols

    size (Int)   = square size in pixels for the resized image
    ------------------------------
    Outputs
    Hashed_Images (Int)= Matrix of size dim01 by size^2 * 2, with the binary representation of each Image
    ------------------------------
    Syntaxis
    D_hash = (Images, size=8)
    '''
    # output format
    format_ = '0' + str(size**2) + 'b'

    #preallocate
    Hashed_Images = _np.zeros((Images.shape[0], size**2 * 2));
    Hashed_Rows = _np.zeros((Images.shape[0], size**2))
    for idx , Img in enumerate(Images):
        row, col = _dhash.dhash_row_col( _PLImage.fromarray(Img) , size = size)
        #transform to binary and concatenate
        hash_ = format(row, format_) + format(col, format_)

        # Allocate the hash values for each image. note: hash_ is a string.
        for colidx,num in enumerate(hash_):
            Hashed_Images[idx,colidx] = int(num)

        # Allocate rows only
        for colidx,num in enumerate( format(col, format_) ):
            Hashed_Rows[idx,colidx] = int(num)

    return Hashed_Images, Hashed_Rows

def classification_metrics(y_true, y_predicted, verbose = 1):
    if verbose == 1:
        print("Classification Report:\n%s" % metrics.classification_report(y_true, y_predicted))
        print(20*'---')
        print("Cohen kappa Score:\n%s" % metrics.cohen_kappa_score(y_true, y_predicted))
        print(20*'---')
        print("Hamming Loss Score:\n%s" % metrics.hamming_loss(y_true, y_predicted))

    kappa = metrics.cohen_kappa_score(y_true, y_predicted)
    hamming_loss = metrics.hamming_loss(y_true, y_predicted)
    accurracy = metrics.accuracy_score(y_true, y_predicted)
    return accurracy, kappa, hamming_loss
