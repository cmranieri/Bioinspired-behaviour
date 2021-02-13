from ExtendedBG import ExtendedBG
import numpy as np
import pickle

t_sim = 70 * 1000

def load_preds( path ):
    with open( path, 'rb' ) as f:
        results = pickle.load( f )
    return results


def get_sequence( results, session ):
    idxs = np.random.choice( 5, 140 ) * 140 + np.arange(140)
    pred = results['predictions'][ session, idxs ]
    lbl  = results['labels'][ session ]
    return pred, lbl


def save( data, out_name ):
    with open( out_name, 'wb' ) as f:
        pickle.dump( data, f )


if __name__ == '__main__':
    for mode in [ 'imu_sh', 'cnn-lstm' ]:
        for fold in range(8):
            print( 'FOLD', fold )
            preds_path = '../preds/model-lyell-%s-0%d.pickle'%(mode,fold+1)
            results = load_preds( preds_path )
            for session in range( results['labels'].shape[0] ):
                preds, lbl = get_sequence( results, session=session )
                print( 'Fold %d, session %d' % (fold, session) )
                print( 'Label:', np.argmax( lbl ) )
                network = ExtendedBG( stim_interval = 500,
                                      t_sim  = t_sim,
                                      has_pd = True,
                                      preds  = preds,
                                      imax   = 1e-3 )
                #network.set_marmoset()
                network.simulate( lfp=False )
                mfr = network.get_mfr( bins=20 )
                network.plot_mfr( mfr )
                #lfp = network.get_lfp()
                print( network._tt )
                save( mfr, '../mfr/rat/mfr_str_pd/%s/fd%d_s%d_lbl%d.pickle' % (mode, fold, session, np.argmax(lbl)) ) 
                #save( lfp, '../lfp/imu_sh/s%d_%d.pickle' % (fold, session) ) 
