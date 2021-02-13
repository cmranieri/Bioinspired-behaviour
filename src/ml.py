import pickle
import numpy as np
import glob
import os
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


def window_data( x, y, window_size, step=None ):
    if step is None:
        step = window_size
    # superposition
    sp = window_size // step
    w_x = [ x[ i * step :  i * step + window_size ]
            for i in range( len(x) // step - sp ) ]
    w_y = [ y for i in range( len(x) // step - sp ) ]
    #print( np.array(w_x).shape )
    return w_x, w_y


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
    

def load_set( fnames, window_size=None, step=None ):
    X = list()
    Y = list()
    for fname in fnames:
        x, y = load_file( fname )
        if window_size is None:
            X.append( x )
            Y.append( y )
        else:
            w_x, w_y = window_data( x, y, window_size, step )
            X += list( w_x )
            Y += list( w_y )
    X = np.array( X, dtype = np.float32 )
    Y = np.array( Y, dtype = np.int16 )
    return X, Y


def load_data( fold, preds_dir, window_size, step ):
    train_fnames = glob.glob( os.path.join( preds_dir, 'fd[!%d]_*'%fold ) )
    test_fnames  = glob.glob( os.path.join( preds_dir, 'fd%d_*'%fold ) )
    X_train, y_train = load_set( train_fnames, window_size, step )
    X_test,  y_test  = load_set( test_fnames,  window_size, step )
    return X_train, X_test, y_train, y_test


def build_model_cnn( shape ):
    x = Input( shape = shape )
    #y = BatchNormalization()(x)
    y = Conv1D( 128, 7, padding = 'same', activation = 'relu' )(x)
    y = MaxPooling1D( 2 )(y)
    y = Conv1D( 256, 7, padding = 'same', activation = 'relu' )(y)
    y = MaxPooling1D( 2 )(y)
    y = GlobalAveragePooling1D()(y)
    y = Dense( 3 )(y)
    y = Activation( 'softmax' )(y)
    model = Model( x, y )
    optimizer = SGD( lr=1e-2, decay=1e-4 )
    model.compile( loss      = 'categorical_crossentropy',
                   optimizer = 'adam',#optimizer,
                   metrics   = [ 'acc' ] )
    return model


def train_model( animal, stim_cond, modalities ):
    preds_dir = os.path.join( '..', 'mfr', animal, stim_cond, modalities )
    print(preds_dir)
    histories = list()
    for fold in range(8):
        print( animal, stim_cond, modalities, fold )
        X_train, X_test, y_train, y_test = load_data( fold, preds_dir, window_size=200, step=50 )
        shape = X_train.shape[1:]

        model = build_model_cnn( shape )
        hist  = model.fit( X_train, y_train,
                           validation_data = (X_test, y_test),
                           batch_size=32,
                           epochs = 40 )
        histories.append( hist.history )
        fname = '%s-%s-%s-f%d.h5' % ( animal, stim_cond, modalities, fold ) 
        model.save( os.path.join( '..', 'models', fname ) )
    acc = [ h['val_acc'][-1] for h in histories ]
    print( 'Mean acc: %0.4f \nStd acc: %0.4f \nAcc: %s' % ( np.mean( acc ), np.std( acc ), acc ) )
    return acc


def train_all():
    for modalities in [ 'imu_sh', 'cnn-lstm' ]:
        for animal in [ 'rat', 'marmoset' ]:
            for stim_cond in [ 'mfr_str', 'mfr_str_pd', 'mfr2_str', 'mfr2_str_pd' ]:
                acc = train_model( animal, stim_cond, modalities )
                with open( 'results.txt', 'a' ) as f:
                    f.write( '%s, %s, %s\n' % (animal, stim_cond, modalities) )
                    f.write( 'Mean acc: %0.4f \nStd acc: %0.4f \nAcc: %s\n\n' % ( np.mean( acc ), np.std( acc ), acc ) )


def predict_model( animal, stim_cond, modalities, fold ):
    preds_dir = os.path.join( '..', 'mfr', animal, stim_cond, modalities )
    X_train, X_test, y_train, y_test = load_data( fold, preds_dir, window_size=None, step=None )
    model_fname = '%s-%s-%s-f%d.h5' % ( animal, stim_cond, modalities, fold ) 
    model = load_model( os.path.join( '..', 'models_w', model_fname ) )
    preds_list = list()
    for i in range( len(X_test) ):
        wx_test, wy_test = window_data( X_test[i], y_test[i], window_size=200, step=50 )
        preds = model.predict( np.array(wx_test) )
        preds_list.append( preds )
    return np.array( preds_list ), y_test


def predict_all():
    for modalities in [ 'imu_sh', 'cnn-lstm' ]:
        for animal in [ 'rat', 'marmoset' ]:
            for stim_cond in [ 'mfr_str', 'mfr_str_pd', 'mfr2_str', 'mfr2_str_pd' ]:
                for fold in range(8):
                    predictions, labels = predict_model( animal, stim_cond, modalities, fold )
                    fname = '%s-%s-%s-f%d.pickle' % ( animal, stim_cond, modalities, fold ) 
                    f = open( os.path.join( '..', 'pred_behs', fname ), 'wb' )
                    pickle.dump( { 'predictions': predictions,
                                   'labels': labels }, f )
                    f.close()




if __name__ == '__main__':
    #train_all()
    predict_all()






