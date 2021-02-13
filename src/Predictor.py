#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import pickle
import numpy as np


class Predictor:
    def __init__( self, preds_path, results_path ):
        self._preds_path = preds_path
        self._results_path = results_path


    def load_preds( self, path ):
        with open( path, 'rb' ) as f:
            results = pickle.load( f )
        return results


    def get_sequence( self, results, act_seq, session ):
        idxs = np.random.choice( 5, 140 ) * 140 + np.arange(140)
        pred = results['predictions'][ act_seq[ session ], idxs ]
        lbl  = results['labels'][ act_seq[ session ] ]
        return pred, lbl


    def predict( self ):
        results = self.load_preds( self._preds_path )
        act_seq = np.random.choice( results['labels'].shape[0], 
                                    results['labels'].shape[0], 
                                    replace=False )
        pub  = rospy.Publisher('prediction', String, queue_size=10)
        pubf = rospy.Publisher('finish_preds', String, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(0.5) # hz
        rospy.sleep(5.)
        for session in range( results['labels'].shape[ 0 ] ):
        #for session in range( 6 ):
            pred, lbl = self.get_sequence( results, act_seq, session )
            for t in range( pred.shape[ 0 ] ):
                if rospy.is_shutdown():
                    return
                msg = np.concatenate( [pred[t], [session], [np.argmax(lbl)] ],
                                      axis=0 )
                hello_str = "session %d; timestep %d; %s" % (
                            session, 
                            t, 
                            rospy.get_time() )
                rospy.loginfo( hello_str )
                pub.publish( str( msg ) )
                rate.sleep()
        pubf.publish( self._results_path )


if __name__ == '__main__':
    for model_name in [ 'imu_sh', 'cnn-lstm' ]:
        for i in range(8):
            print( 'MODEL', model_name )
            print( 'FOLD', i )
            preds_path = '../data/model-lyell-%s-0%d.pickle'%(model_name,i+1)
            results_path = '../results/sim_%s_%d.pickle'%(model_name,i+1)
            try:
                predictor = Predictor( preds_path, results_path )
                predictor.predict()
            except rospy.ROSInterruptException:
                pass
