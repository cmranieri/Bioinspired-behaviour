from ExtendedBG import ExtendedBG
import numpy as np
import pickle

t_sim = 20 * 1000

def load_preds( path ):
    with open( path, 'rb' ) as f:
        results = pickle.load( f )
    return results


def get_sequence( results, session ):
    idxs = np.random.choice( 5, 140 ) * 140 + np.arange(140)
    pred = results['predictions'][ session, idxs ]
    lbl  = results['labels'][ session ]
    return pred, lbl


if __name__ == '__main__':
    fold = 2
    preds_path = '../data/model-lyell-imu_sh-0%d.pickle'%(fold+1)

    results = load_preds( preds_path )
    preds, lbl = get_sequence( results, session=15 )
    print( 'Label:', np.argmax( lbl ) )
 
    network = ExtendedBG( stim_interval = 500,
                          t_sim = t_sim,
                          has_pd = False,
                          preds = preds )
    network.simulate( recordStep=10, lfp=True )
    mfr = network.get_mfr( bins=10 )
    lfp = network.get_lfp()
 
