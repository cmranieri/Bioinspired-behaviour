import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score


for animal in [ 'rat', 'marmoset' ]:
    for stim in [ 'mfr_str', 'mfr2_str' ]:
        for modalities in [ 'imu_sh', 'cnn-lstm' ]:
            for cond in [ '', '_pd' ]:
                acc_list = list()
                f1_list = list()
                for fold in range( 8 ):
                    path = '../decisions/%s-%s%s-%s-f%d.pickle' % ( animal, stim, cond, modalities, fold )

                    with open( path, 'rb' ) as f:
                        decisions = pickle.load( f )
                    preds = decisions['predictions']
                    lbl   = decisions['labels']
                    lbl2 = np.array( [ [list(lbl[i])] * preds.shape[1] for i in range( lbl.shape[0] ) ] )

                    y_pred = np.reshape( preds, [-1, 3] ).argmax(1)
                    y_true = np.reshape( lbl2,  [-1, 3] ).argmax(1)

                    acc_list.append( accuracy_score( y_true, y_pred ) )
                    f1_list.append(  f1_score( y_true, y_pred, average='macro' ) )

                print( '%f, %f, %f, %f' % ( np.mean( acc_list ) * 100,
                                            np.std(  acc_list ) * 100,
                                            np.mean( f1_list)   * 100,
                                            np.std( f1_list )   * 100 ) )
