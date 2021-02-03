import pickle
import numpy as np
import glob
import re
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter

from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras.backend import set_session
config = ConfigProto()
config.gpu_options.allow_growth=True
sess = Session(config=config)
set_session( sess )


def normalize( X ):
    scaler = StandardScaler()
    X = scaler.fit_transform( X )
    return X


def class2behaviour( lbl ):
    lbl = int( lbl )
    if lbl in [ 0, 4, 7 ]:
        y = [ 1, 0, 0 ]
    elif lbl in [ 2, 3, 5 ]:
        y = [ 0, 1, 0 ]
    else:
        y = [ 0, 0, 1 ]
    return y


def load_file( path ):
    lbl = re.match( '.*lbl(\d).pickle', path ).groups()[0]
    y = class2behaviour( lbl )
    with open( path, 'rb' ) as f:
        x = pickle.load( f )
    x = normalize( x )
    # [ b, t, f ]
    x = np.transpose( x, [1, 0] )
    # only cortex neurons
    x = x[:,:4]
    return x, y
    

def load_set( fnames ):
    X = list()
    Y = list()
    for fname in fnames:
        x, y = load_file( fname )
        X.append( x )
        Y.append( y )
    X = np.array( X, dtype = np.float32 )
    Y = np.array( Y, dtype = np.int16 )
    return X, Y


def load_data( fold ):
    train_fnames = glob.glob( '../mfr/imu_sh/fd[!%d]_*'%fold )
    test_fnames  = glob.glob( '../mfr/imu_sh/fd%d_*'%fold )
    X_train, y_train = load_set( train_fnames )
    X_test,  y_test  = load_set( test_fnames )
    return X_train, X_test, y_train, y_test


def build_model_cnn( shape ):
    x = Input( shape = shape )
    y = BatchNormalization()(x)
    y = Conv1D( 128, 5, padding = 'same', activation = 'relu' )(y)
    y = MaxPooling1D( 2 )(y)
    y = Conv1D( 256, 5, padding = 'same', activation = 'relu' )(y)
    y = MaxPooling1D( 2 )(y)
    y = GlobalAveragePooling1D()(y)
    #y = LSTM( 128, return_sequences=False )(y)
    y = Dense( 3 )(y)
    y = Activation( 'softmax' )(y)
    model = Model( x, y )
    optimizer = SGD( lr=1e-2, decay=1e-4 )
    model.compile( loss      = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics   = [ 'acc' ] )
    return model


if __name__ == '__main__':
    histories = list()
    for fold in range(8):
        X_train, X_test, y_train, y_test = load_data( fold )
        shape = X_train.shape[1:]

        model = build_model_cnn( shape )
        hist  = model.fit( X_train, y_train,
                           validation_data = (X_test, y_test),
                           batch_size=32,
                           epochs = 70 )
        histories.append( hist.history )
    acc = [ h['val_acc'][-1] for h in histories ]
    print( 'Mean acc: %0.4f \nStd acc: %0.4f \nAcc: %s' % ( np.mean( acc ), np.std( acc ), acc ) )




