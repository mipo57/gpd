#include <algorithm>
#include <vector>
#include <mutex>

// ROS
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// PCL
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// GPG
#include <gpg/cloud_camera.h>

// this project (messages)
#include <gpd/CloudIndexed.h>
#include <gpd/CloudSamples.h>
#include <gpd/CloudSources.h>
#include <gpd/GraspConfig.h>
#include <gpd/GraspConfigList.h>
#include <gpd/SamplesMsg.h>
#include <gpd/FindGrasps.h>

// this project (headers)
#include "../../include/gpd/grasp_detector.h"
#include "../../include/gpd/sequential_importance_sampling.h"

#include <gpd/SetParameters.h>

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudPointNormal;


/** GraspDetectionNode class
 *
 * \brief A ROS node that can detect grasp poses in a point cloud.
 *
 * This class is a ROS node that handles all the ROS topics.
 *
*/
class GraspDetectionNode
{
public:

  /**
   * \brief Constructor.
   * \param node the ROS node
  */
  GraspDetectionNode(ros::NodeHandle& node);

  /**
   * \brief Destructor.
  */
  ~GraspDetectionNode()
  {
    delete cloud_camera_;

    if (use_importance_sampling_)
    {
      delete importance_sampling_;
    }

    delete grasp_detector_;
  }

  /**
   * \brief Run the ROS node. Loops while waiting for incoming ROS messages.
  */
  void run();

  /**
   * \brief Detect grasp poses in a point cloud received from a ROS topic.
   * \return the list of grasp poses
  */
  std::vector<Grasp> detectGraspPosesInTopic();


private:

  /**
   * \brief Find the indices of the points within a ball around a given point in the cloud.
   * \param cloud the point cloud
   * \param centroid the centroid of the ball
   * \param radius the radius of the ball
   * \return the indices of the points in the point cloud that lie within the ball
  */
  std::vector<int> getSamplesInBall(const PointCloudRGBA::Ptr& cloud, const pcl::PointXYZRGBA& centroid, float radius);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud.
   * \param cloud cloud on which detection will be made
  */
  void set_cloud(const sensor_msgs::PointCloud2& cloud);

  /**
   * \brief Callback function for the ROS find grasps service query.
   * \param req incoming data
   * \param resp resuponse data
  */
  bool find_graps_callback(gpd::FindGrasps::Request& req, gpd::FindGrasps::Response& resp);

  /**
   * \brief Initialize the <cloud_camera> object given a <cloud_sources> message.
   * \param msg the <cloud_sources> message
   */
  void initCloudCamera(const gpd::CloudSources& msg);

  /**
   * \brief Create a ROS message that contains a list of grasp poses from a list of handles.
   * \param hands the list of grasps
   * \return the ROS message that contains the grasp poses
  */
  gpd::GraspConfigList createGraspListMsg(const std::vector<Grasp>& hands);

  gpd::GraspConfig convertToGraspMsg(const Grasp& hand);

  visualization_msgs::MarkerArray convertToVisualGraspMsg(const std::vector<Grasp>& hands, double outer_diameter,
    double hand_depth, double finger_width, double hand_height, const std::string& frame_id);

  visualization_msgs::Marker createFingerMarker(const Eigen::Vector3d& center, const Eigen::Matrix3d& frame,
    double length, double width, double height, int id, const std::string& frame_id);

  visualization_msgs::Marker createHandBaseMarker(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
      const Eigen::Matrix3d& frame, double length, double height, int id, const std::string& frame_id);

  Eigen::Matrix3Xd fillMatrixFromFile(const std::string& filename, int num_normals);

  Eigen::Vector3d view_point_; ///< (input) view point of the camera onto the point cloud

  CloudCamera* cloud_camera_; ///< stores point cloud with (optional) camera information and surface normals
  std_msgs::Header cloud_camera_header_; ///< stores header of the point cloud

  int size_left_cloud_; ///< (input) size of the left point cloud (when using two point clouds as input)
  bool has_cloud_, has_normals_, has_samples_; ///< status variables for received (input) messages
  std::string frame_; ///< point cloud frame

  ros::NodeHandle nh_; ///< ROS node handle
  ros::Subscriber cloud_sub_; ///< ROS subscriber for point cloud messages
  ros::Subscriber samples_sub_; ///< ROS subscriber for samples messages
  ros::Publisher grasps_pub_; ///< ROS publisher for grasp list messages
  ros::Publisher grasps_rviz_pub_; ///< ROS publisher for grasps in rviz (visualization)
  ros::ServiceServer srv_set_params_; ///< ROS service server for setting params
  ros::ServiceServer srv_find_graps;

  bool use_importance_sampling_; ///< if importance sampling is used
  bool filter_grasps_; ///< if grasps are filtered on workspace and gripper aperture
  bool filter_half_antipodal_; ///< if half-antipodal grasps are filtered
  bool plot_filtered_grasps_; ///< if filtered grasps are plotted
  bool plot_selected_grasps_; ///< if selected grasps are plotted
  bool plot_normals_; ///< if normals are plotted
  bool plot_samples_; ///< if samples/indices are plotted
  bool use_rviz_; ///< if rviz is used for visualization instead of PCL
  std::vector<double> workspace_; ///< workspace limits
  std::vector<Grasp> grasps_; /// Most recent graphs
  std::mutex graps_mutex_;

  GraspDetector* grasp_detector_; ///< used to run the grasp pose detection
  SequentialImportanceSampling* importance_sampling_; ///< sequential importance sampling variation of grasp pose detection

  /** constants for input point cloud types */
  static const int POINT_CLOUD_2; ///< sensor_msgs/PointCloud2
  static const int CLOUD_INDEXED; ///< gpd/CloudIndexed
  static const int CLOUD_SAMPLES; ///< gpd/CloudSamples
};

/** constants for input point cloud types */
const int GraspDetectionNode::POINT_CLOUD_2 = 0; ///< sensor_msgs/PointCloud2
const int GraspDetectionNode::CLOUD_INDEXED = 1; ///< cloud with indices
const int GraspDetectionNode::CLOUD_SAMPLES = 2; ///< cloud with (x,y,z) samples


GraspDetectionNode::GraspDetectionNode(ros::NodeHandle& node) : has_cloud_(false), has_normals_(false),
  size_left_cloud_(0), has_samples_(true), frame_("")
{
  cloud_camera_ = NULL;

  nh_ = node; // Assign the NodeHandle to the private variable

  // set camera viewpoint to default origin
  std::vector<double> camera_position;
  nh_.getParam("camera_position", camera_position);
  view_point_ << camera_position[0], camera_position[1], camera_position[2];

  // choose sampling method for grasp detection
  nh_.param("use_importance_sampling", use_importance_sampling_, false);

  if (use_importance_sampling_)
  {
    importance_sampling_ = new SequentialImportanceSampling(nh_);
  }
  grasp_detector_ = new GraspDetector(nh_);

  // Read input cloud and sample ROS topics parameters.
  int cloud_type;
  nh_.param("cloud_type", cloud_type, POINT_CLOUD_2);
  std::string cloud_topic;
  nh_.param("cloud_topic", cloud_topic, std::string("/camera/depth_registered/points"));
  std::string rviz_topic;
  nh_.param("rviz_topic", rviz_topic, std::string(""));

  if (!rviz_topic.empty())
  {
    grasps_rviz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(rviz_topic, 1);
    use_rviz_ = true;
  }
  else
  {
    use_rviz_ = false;
  }

  srv_find_graps = nh_.advertiseService("find_graps", &GraspDetectionNode::find_graps_callback, this);

  nh_.getParam("workspace", workspace_);
}


void GraspDetectionNode::run()
{
  ros::Rate rate(5);
  ROS_INFO("Waiting for point cloud to arrive ...");

  while (ros::ok())
  {
    if (has_cloud_)
    {
      // detect grasps in point cloud
      // std::vector<Grasp> grasps = detectGraspPosesInTopic();

      // visualize grasps in rviz
      if (use_rviz_)
      {
        std::lock_guard<std::mutex> lock(graps_mutex_);
        
        const HandSearch::Parameters& params = grasp_detector_->getHandSearchParameters();
        grasps_rviz_pub_.publish(convertToVisualGraspMsg(this->grasps_, params.hand_outer_diameter_, params.hand_depth_,
                                                         params.finger_width_, params.hand_height_, frame_));
      }

      // reset the system
      // has_cloud_ = false;
      // has_samples_ = false;
      // has_normals_ = false;
    }

    ros::spinOnce();
    rate.sleep();
  }
}

std::vector<Grasp> GraspDetectionNode::detectGraspPosesInTopic()
{
  // detect grasp poses
  std::vector<Grasp> grasps;

  if (use_importance_sampling_)
  {
    cloud_camera_->filterWorkspace(workspace_);
    cloud_camera_->voxelizeCloud(0.003);
    cloud_camera_->calculateNormals(4);
    grasps = importance_sampling_->detectGrasps(*cloud_camera_);
  }
  else
  {
    // preprocess the point cloud
    grasp_detector_->preprocessPointCloud(*cloud_camera_);

    // detect grasps in the point cloud
    grasps = grasp_detector_->detectGrasps(*cloud_camera_);
  }

  // Publish the selected grasps.
  gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
  grasps_pub_.publish(selected_grasps_msg);
  ROS_INFO_STREAM("Published " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");

  return grasps;
}


std::vector<int> GraspDetectionNode::getSamplesInBall(const PointCloudRGBA::Ptr& cloud,
  const pcl::PointXYZRGBA& centroid, float radius)
{
  std::vector<int> indices;
  std::vector<float> dists;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  kdtree.radiusSearch(centroid, radius, indices, dists);
  return indices;
}


void GraspDetectionNode::set_cloud(const sensor_msgs::PointCloud2& msg)
{
  if (cloud_camera_)
    delete cloud_camera_;

  cloud_camera_ = NULL;

  Eigen::Matrix3Xd view_points(3,1);
  view_points.col(0) = view_point_;

  if (msg.fields.size() == 6 && msg.fields[3].name == "normal_x" && msg.fields[4].name == "normal_y"
    && msg.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(msg, *cloud);
    cloud_camera_ = new CloudCamera(cloud, 0, view_points);
    cloud_camera_header_ = msg.header;
    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points and normals.");
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(msg, *cloud);
    cloud_camera_ = new CloudCamera(cloud, 0, view_points);
    cloud_camera_header_ = msg.header;
    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points.");
  }

  has_cloud_ = true;
  frame_ = msg.header.frame_id;
  
}

bool GraspDetectionNode::find_graps_callback(gpd::FindGrasps::Request& req, gpd::FindGrasps::Response& resp)
{
  std::lock_guard<std::mutex> lock(graps_mutex_);

  this->set_cloud(req.cloud);
  this->grasps_ = this->detectGraspPosesInTopic();

  resp.grasps = createGraspListMsg(this->grasps_);

  return true;
}

void GraspDetectionNode::initCloudCamera(const gpd::CloudSources& msg)
{
  // clean up
  delete cloud_camera_;
  cloud_camera_ = NULL;

  // Set view points.
  Eigen::Matrix3Xd view_points(3, msg.view_points.size());
  for (int i = 0; i < msg.view_points.size(); i++)
  {
    view_points.col(i) << msg.view_points[i].x, msg.view_points[i].y, msg.view_points[i].z;
  }

  // Set point cloud.
  if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x"
    && msg.cloud.fields[4].name == "normal_y" && msg.cloud.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
    std::cout << "view_points:\n" << view_points << "\n";
  }
}


gpd::GraspConfigList GraspDetectionNode::createGraspListMsg(const std::vector<Grasp>& hands)
{
  gpd::GraspConfigList msg;

  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(convertToGraspMsg(hands[i]));

  msg.header = cloud_camera_header_;

  return msg;
}


gpd::GraspConfig GraspDetectionNode::convertToGraspMsg(const Grasp& hand)
{
  gpd::GraspConfig msg;
  tf::pointEigenToMsg(hand.getGraspBottom(), msg.bottom);
  tf::pointEigenToMsg(hand.getGraspTop(), msg.top);
  tf::pointEigenToMsg(hand.getGraspSurface(), msg.surface);
  tf::vectorEigenToMsg(hand.getApproach(), msg.approach);
  tf::vectorEigenToMsg(hand.getBinormal(), msg.binormal);
  tf::vectorEigenToMsg(hand.getAxis(), msg.axis);
  msg.width.data = hand.getGraspWidth();
  msg.score.data = hand.getScore();
  tf::pointEigenToMsg(hand.getSample(), msg.sample);

  return msg;
}


visualization_msgs::MarkerArray GraspDetectionNode::convertToVisualGraspMsg(const std::vector<Grasp>& hands,
  double outer_diameter, double hand_depth, double finger_width, double hand_height, const std::string& frame_id)
{
  double width = outer_diameter;
  double hw = 0.5 * width;

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker left_finger, right_finger, base, approach;
  Eigen::Vector3d left_bottom, right_bottom, left_top, right_top, left_center, right_center, approach_center,
    base_center;

  for (int i = 0; i < hands.size(); i++)
  {
    left_bottom = hands[i].getGraspBottom() - (hw - 0.5*finger_width) * hands[i].getBinormal();
    right_bottom = hands[i].getGraspBottom() + (hw - 0.5*finger_width) * hands[i].getBinormal();
    left_top = left_bottom + hand_depth * hands[i].getApproach();
    right_top = right_bottom + hand_depth * hands[i].getApproach();
    left_center = left_bottom + 0.5*(left_top - left_bottom);
    right_center = right_bottom + 0.5*(right_top - right_bottom);
    base_center = left_bottom + 0.5*(right_bottom - left_bottom) - 0.01*hands[i].getApproach();
    approach_center = base_center - 0.04*hands[i].getApproach();

    base = createHandBaseMarker(left_bottom, right_bottom, hands[i].getFrame(), 0.02, hand_height, i, frame_id);
    left_finger = createFingerMarker(left_center, hands[i].getFrame(), hand_depth, finger_width, hand_height, i*3, frame_id);
    right_finger = createFingerMarker(right_center, hands[i].getFrame(), hand_depth, finger_width, hand_height, i*3+1, frame_id);
    approach = createFingerMarker(approach_center, hands[i].getFrame(), 0.08, finger_width, hand_height, i*3+2, frame_id);

    marker_array.markers.push_back(left_finger);
    marker_array.markers.push_back(right_finger);
    marker_array.markers.push_back(approach);
    marker_array.markers.push_back(base);
  }

  return marker_array;
}


visualization_msgs::Marker GraspDetectionNode::createFingerMarker(const Eigen::Vector3d& center,
  const Eigen::Matrix3d& frame, double length, double width, double height, int id, const std::string& frame_id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time();
  marker.ns = "finger";
  marker.id = id;
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = center(0);
  marker.pose.position.y = center(1);
  marker.pose.position.z = center(2);
  marker.lifetime = ros::Duration(10);

  // use orientation of hand frame
  Eigen::Quaterniond quat(frame);
  marker.pose.orientation.x = quat.x();
  marker.pose.orientation.y = quat.y();
  marker.pose.orientation.z = quat.z();
  marker.pose.orientation.w = quat.w();

  // these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length; // forward direction
  marker.scale.y = width; // hand closing direction
  marker.scale.z = height; // hand vertical direction

  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.5;

  return marker;
}


visualization_msgs::Marker GraspDetectionNode::createHandBaseMarker(const Eigen::Vector3d& start,
  const Eigen::Vector3d& end, const Eigen::Matrix3d& frame, double length, double height, int id,
  const std::string& frame_id)
{
  Eigen::Vector3d center = start + 0.5 * (end - start);

  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time();
  marker.ns = "hand_base";
  marker.id = id;
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = center(0);
  marker.pose.position.y = center(1);
  marker.pose.position.z = center(2);
  marker.lifetime = ros::Duration(10);

  // use orientation of hand frame
  Eigen::Quaterniond quat(frame);
  marker.pose.orientation.x = quat.x();
  marker.pose.orientation.y = quat.y();
  marker.pose.orientation.z = quat.z();
  marker.pose.orientation.w = quat.w();

  // these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length; // forward direction
  marker.scale.y = (end - start).norm(); // hand closing direction
  marker.scale.z = height; // hand vertical direction

  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 1.0;

  return marker;
}

bool detect_graps(gpd::FindGrasps::Request& req, gpd::FindGrasps::Request& resp)
{
  

}

int main(int argc, char** argv)
{
  // seed the random number generator
  std::srand(std::time(0));

  // initialize ROS
  ros::init(argc, argv, "detect_grasps");
  ros::NodeHandle node("~");

  //node.advertiseService("find_graps", )

  GraspDetectionNode grasp_detection(node);
  grasp_detection.run();

  return 0;
}
