import rospy
import actionlib
import time
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from math import radians

class Navigate():
    def __init__(self):
        #rospy.init_node('navgoals_py')
        self._client = actionlib.SimpleActionClient('move_base',
                                                    MoveBaseAction)
        self.locations = { 'shelf1': [ -0.4,  3.4, 0 ],
                           'shelf2': [ -0.4,  0.5, 0 ],
                            'table': [ -0.8, -3.5, 180 ],
                            'sink':  [  4.0, -0.7, 270 ] }
        self.goal_location = None


    def start_move(self, location):
        self.goal_location = location
        self._client.wait_for_server()
        goal = self.setupGoal( location )
        self._client.send_goal( goal )
        return

    def stop_move( self ):
        self.goal_location = None
        self._client.cancel_all_goals()
        return

    def setupGoal(self, location):
        x, y, yaw = self.locations[ location ]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0
        goal.target_pose.pose.position.z = 0.0
        angle = radians(yaw)  # angles are expressed in radians
        quaternion = quaternion_from_euler(0.0, 0.0, angle)  # roll, pitch, yaw
        goal.target_pose.pose.orientation = Quaternion(*quaternion.tolist())
        return goal


if __name__ == '__main__':
    userdata = dict()

    navigate = Navigate()
    navigate.start_move( 'sink' )
