from NavGoals import Navigate
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseActionResult
import rospy
import re
import numpy as np
import pickle

class Robot:
    def __init__( self ):
        self.EMPTY = 0
        self.OBJ1  = 1
        self.OBJ2  = 2
        self.OBJ3  = 3

        self._pred_action = { 0: 'B1', 
                              1: 'B2', 
                              2: 'B3' } 
        self._navigate  = Navigate()
        self._is_moving = False
        self._reset()


    def _reset( self ):
        self._current_action  = None
        self._action_finished = False
        self._position        = None
        self._carry           = self.EMPTY
        self._session         = None
        self._lbl             = None
        self._wpred           = None
        self._time_start      = None
        self._results         = list()
        self._times_elapsed   = list()
        self._action_history  = list()
 

    def _set_action( self, action ):
        print( 'session: %s, action: %s, carry: %s, label: %s, expected: %s'%(
                self._session,
                action,
                self._carry, 
                self._lbl, 
                self._pred_action[self._lbl]) )
        self._action_history.append( [ self._session, self._lbl, self._pred_action[ self._lbl ], action ] )
        if action != self._current_action:
            self._navigate.stop_move()
            self._current_action = action
            #self._is_moving = False
        if action == 'B1':
            self.obj1_shelf1_table()
        elif action == 'B2':
            self.obj2_shelf2_table()
        elif action == 'B3':
            #self.obj3_shelf1_table()
            self._return_obj()
        return


    def start_move( self, dstpos ):
        self._navigate.start_move( dstpos )
        self._is_moving = True


    # Scheduler
    def predsCallback( self, data ):
        #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        data_arr = np.array( data.data.strip( '][' ).split(),
                             dtype=np.float32 )
        # check if session is over
        session  = int( data_arr[ -2 ] )
        new_lbl  = int( data_arr[ -1 ] )
        if session != self._session:
            self.evaluate_result()
            self._session = session
            self._lbl     = new_lbl
            self._wpred   = None
            self._action_finished = False
            self._time_start = rospy.get_time()
        # check if action has finished
        elif self._action_finished:
            print( 'ACTION FINISHED' )
            return
        # update action according to prediction
        pred = np.array( data_arr[ :-2 ], dtype=np.float32 )
        self.update_wpred( pred )
        new_action = self._pred_action[ np.argmax( self._wpred ) ]
        self._set_action( new_action )


    def finishCallback( self, data ):
        path = data.data
        self.evaluate_result()
        res_dict = dict()
        res_dict[ 'times']    = self._times_elapsed
        res_dict[ 'history' ] = self._action_history
        res_dict[ 'results' ] = self._results
        with open( path, 'wb' ) as f:
            pickle.dump( res_dict, f )
        self._reset()


    def goalCallback( self, data ):
        self._is_moving = False
        fnd = re.findall( 'Goal reached', str(data) )
        if fnd != []:
            self._position = self._navigate.goal_location
            print( 'Position: %s'%self._position )


    def update_wpred( self, pred ):
        if self._wpred is None:
            self._wpred = pred
        else:
            self._wpred = 0.9*self._wpred + pred
        return self._wpred


    def evaluate_result( self ):
        if self._lbl is not None:
            correct_action = self._pred_action[ self._lbl ]
            if self._action_finished:
                if self._current_action == correct_action:
                    print( 'Finished correctly!' )
                    self._results.append( 'correct' )
                else:
                    print( 'Finished incorrectly!' )
                    self._results.append( 'incorrect' )
            else:
                if correct_action=='B3':
                    print( 'Finished correctly (no action required)!' )
                    self._results.append( 'correct' )
                else:
                    print( 'Did not finish!' )
                    self._results.append( 'unfinished' )
                self._set_action_finished()


    def _update_carry( self, new_item ):
        self._carry = new_item
        self._position = None


    def _return_obj( self ):
        print( 'RETURN OBJ' )
        # Return OBJ1 to shelf1
        if self._carry == self.OBJ1:
            if self._position == 'shelf1':
                self._update_carry( self.EMPTY )
            else:
                self.start_move( 'shelf1' )
        # Return OBJ2 to shelf2
        elif self._carry == self.OBJ2:
            if self._position == 'shelf2':
                self._update_carry( self.EMPTY )
            else:
                self.start_move( 'shelf2' )
        # Return OBJ3 to shelf1
        elif self._carry == self.OBJ3:
            if self._position == 'shelf1':
                self._update_carry( self.EMPTY )
            else:
                self.start_move( 'shelf1' )


    def _set_action_finished( self ):
        self._update_carry( self.EMPTY )
        self._action_finished = True
        time_elapsed = rospy.get_time() - self._time_start
        self._times_elapsed.append( time_elapsed )


    def move_obj( self, obj, srcpos, dstpos ):
        #if not self._is_moving:
            # Finished action?
            if self._carry == obj and self._position == dstpos:
                self._set_action_finished()
            # Needs to get the object?
            elif self._carry == self.EMPTY:
                # Is next to the object?
                if self._position == srcpos:
                    self._update_carry( obj )
                    self.start_move( dstpos )
                # Needs to navigate to the object?
                else:
                    self.start_move( srcpos )
            # Has the correct object?
            elif self._carry == obj:
                self.start_move( dstpos )
            # Has the wrong object and needs to return it?
            else:
                self._return_obj()


    # B1
    def obj1_shelf1_table( self ):
        self.move_obj( self.OBJ1, 'shelf1', 'table' )
    # B2
    def obj2_shelf2_table( self ):
        self.move_obj( self.OBJ2, 'shelf2', 'table' )
    # B3
    #def obj3_shelf1_table( self ):
    #    self.move_obj( self.OBJ3, 'shelf1', 'table' )

    
    def listener( self ):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber( '/move_base/result', 
                          MoveBaseActionResult,
                          self.goalCallback)
        rospy.Subscriber( '/prediction',
                          String,
                          self.predsCallback)
        rospy.Subscriber( '/finish_preds',
                          String,
                          self.finishCallback)
        rospy.spin()


if __name__ == '__main__':
    robot = Robot()
    robot.listener()
