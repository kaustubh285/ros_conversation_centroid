#!/usr/bin/env python3
"""
Citations:
Facebook AI Research (2021). Detectron.
Available at: https://github.com/facebookresearch/Detectron (Accessed: 27 June 2023).

Kumar Sambhav 2019. mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt. [online] Available at: https://github.com/sambhav37/Mask-R-CNN/blob/master/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt [29 July 2023]
"""

import rospy
import cv2
import traceback
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import numpy as np
import json


class ConversationDetectionNode:
    def __init__(self) -> None:
        # This line lets us know when the node has begun executing
        rospy.init_node("conversation_detection_node")
        rospy.loginfo("Inside __init__")
        # Static
        # The following values were pre-calculated for the test-cases
        (
            self.color_fx,
            self.color_fy,
            self.color_cx,
            self.color_cy,
            self.depth_fx,
            self.depth_fy,
            self.depth_cx,
            self.depth_cy,
        ) = (
            605.2489624023438,
            603.880859375,
            323.78955078125,
            246.0689697265625,
            426.9704895019531,
            426.9704895019531,
            423.92333984375,
            233.7213134765625,
        )
        self.bridge = CvBridge()

        # These values need to be updated to provide the .pb and .pbtxt files for the pretrained model
        pb_file_path = "/home/kausubhd/catkin_ws/src/nav_human_communication/src/scripts/resources/frozen_inference_graph.pb"
        pbtxt_file_path = "/home/kausubhd/catkin_ws/src/nav_human_communication/src/scripts/resources/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

        self.pub = rospy.Publisher(
            "/custom/nav/human_conversation_detection", String, queue_size=20
        )

        self.rgb_image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_image_callback
        )

        self.depth_image_sub = rospy.Subscriber(
            "/camera/depth/image_rect_raw", Image, self.depth_image_callback
        )

        # Dynamic variables
        # This net is later used for object detection
        self.net = cv2.dnn.readNetFromTensorflow(pb_file_path, pbtxt_file_path)
        self.tracked_humans = {}  # An object to keep a track of the humanss
        self.humans_data = []  # All humans detected are first appened to this list
        self.full_depth_img = []
        # A full_depth_img variable which stores the matrix of the depth image to be used at a later point
        self.frame_count = 0
        self.tracker_failure_count = {}

        # The entry and exit logs help us identify any issues that occur during execution
        rospy.loginfo("Exiting __init__")

    def rgb_image_callback(self, msg: Image):
        """
        This callback function is executed whenever the rgb image subscriber receives data, i.e an rgb image
        """
        rospy.loginfo("inside rgb_image_callback")
        try:
            # Process the image and rotate it by 90deg because of camera angle on the robot
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)

            self.rgb_image_main(rgb_image)
            rospy.loginfo("Exiting rgb_img_callback")

        except:
            rospy.logdebug(traceback.print_exc())
            rospy.loginfo("Error rgb_img_callback")

    def rgb_image_main(self, rgb_image):
        """
        This main function will handle the functionalities. It will handle all the function calls
        """
        rospy.loginfo("inside rgb_image_main")
        try:
            self.helper_rgb_human_detect(
                rgb_image
            )  # this will update the self.humans_data variable
            self.update_tracked_humans_with_detected()  # Update self.tracked_humans with new bounding boxes
            self.helper_rgb_human_tracker(
                rgb_image
            )  # this will update the self.tracked_humans variable to add bounding_box and centroid
            self.helper_find_angle_centroid()  # this will also update the self.tracked_humans variable to add the body_angle_x and body_angle_y

            (
                social_interactions,
                interacting_human_indices,
            ) = self.helper_detect_social_interactions()
            # This helper function will handle the interaction detection and return 1) list of strings of interacting humans, and 2) Indices of the interacting humans

            (
                conv_centroid_2d,
                conv_centroid_3d,
            ) = self.calculate_conversation_centroid_2D_3D(interacting_human_indices)

            self.helper_publish_data(
                conv_centroid_2d, conv_centroid_3d, social_interactions
            )

            # This function is called to display the images. It is solely for testing. It can be commented out during deployment phase.
            self.helper_update_display_images(
                "RGB Image",
                rgb_image,
                interacting_human_indices,
                conversation_center_2d=conv_centroid_2d,
                conversation_center_3d=conv_centroid_3d,
            )

            rospy.loginfo("exiting rgb_image_main")

        except:
            rospy.logdebug(traceback.print_exc())
            rospy.loginfo("Error in rgb_image_main")

    def depth_image_callback(self, msg):
        try:
            # Process the image and rotate it by 90deg because of camera angle on the robot
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
            self.full_depth_img = depth_image

        except:
            rospy.logdebug("Error in depth_image_callback")

    def helper_publish_data(
        self, conv_centroid_2d, conv_centroid_3d, social_interactions
    ):
        try:
            rospy.loginfo("Inside publish data")
            data_to_publish = []

            # Create a dictionary for each conversation
            conversation_data = {
                "conversation_centroid_2d": conv_centroid_2d,
                "conversation_centroid_3d": conv_centroid_3d,
                "interacting_humans_type_list": social_interactions,
                "agents_poses": [],
            }

            for idx, human in enumerate(list(self.tracked_humans.items())):
                human = human[1]
                agent_pose = {
                    "agent": f"human-{idx}",
                    "data": {
                        "bounding_box": human["data"]["bounding_box"],
                        "body_angle_x": human["data"]["body_angle_x"],
                        "body_angle_y": human["data"]["body_angle_y"],
                        "centroid_2d": human["data"]["centroid_2d"],
                        "centroid_3d": human["data"]["centroid_3d"].tolist(),
                    },
                }
                conversation_data["agents_poses"].append(agent_pose)

            data_to_publish.append({"conversation_data_0": conversation_data})

            json_string = json.dumps(data_to_publish)

            json_msg = String()
            json_msg.data = json_string

            self.pub.publish(json_msg)

            rospy.loginfo(data_to_publish)
        except:
            rospy.logdebug(traceback.print_exc())
            rospy.loginfo("Error in publish data")

    def helper_rgb_human_detect(self, img):
        """
        Responsible for detecting objects
        """
        rospy.loginfo("Starting helper_rgb_human_detect")
        height, width, _ = img.shape

        # We first provide the rgb image blob to the net, and it then provides us the masks, and bounding boxes
        self.net.setInput(
            cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
        )

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        local_humans = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]

            if (
                score < 0.9 or class_id > 0
            ):  # Filter out objects with low score and non-human class_id
                continue

            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            # Calculate the centroid of the bounding box
            centroid_x = x + (x2 - x) // 2
            centroid_y = y + (y2 - y) // 2

            human_data = {
                "bounding_box": (x, y, x2 - x, y2 - y),
                "centroid_2d": (centroid_x, centroid_y),
            }
            local_humans.append(human_data)

            # TODO : Append local_humans to self.humans_data based on the current frame.
            self.humans_data = local_humans
        rospy.loginfo("Exiting helper_rgb_human_detect")

    def helper_rgb_human_tracker(self, rgb_img):
        rospy.loginfo("inside helper_rgb_human_tracker")
        # Update object trackers for each tracked human in the current frame
        for idx, tracked_human in list(
            self.tracked_humans.items()
        ):  # Use list() to avoid dictionary size change errors
            tracker = tracked_human["tracker"]
            ok, bbox = tracker.update(rgb_img)

            if ok and idx < len(self.humans_data):
                x, y, w, h = self.humans_data[idx]["bounding_box"]
                self.tracked_humans[idx]["data"]["bounding_box"] = (x, y, w, h)
                self.tracked_humans[idx]["data"]["centroid"] = (x + w // 2, y + h // 2)
                self.tracker_failure_count[
                    idx
                ] = 0  # Reset failure count on successful update

            else:
                self.tracker_failure_count[idx] = (
                    self.tracker_failure_count.get(idx, 0) + 1
                )
                if self.tracker_failure_count[idx] > 5:
                    rospy.loginfo(
                        f"Tracker for Human {idx} failed multiple times. Resetting tracker."
                    )
                    del self.tracked_humans[idx]  # Remove the failing tracker
                    del self.tracker_failure_count[
                        idx
                    ]  # Remove failure count for this tracker

        # Initialize new trackers for newly detected humans
        for idx, human in enumerate(self.humans_data):
            if idx not in self.tracked_humans:
                x, y, w, h = human["bounding_box"]
                x = max(0, min(x, rgb_img.shape[1] - 1))
                y = max(0, min(y, rgb_img.shape[0] - 1))
                w = max(1, min(w, rgb_img.shape[1] - x))
                h = max(1, min(h, rgb_img.shape[0] - y))

                bbox = (x, y, w, h)
                tracker = cv2.TrackerMIL_create()
                tracker.init(rgb_img, bbox)
                human["frame_index"] = 0  # Initialize the frame index
                self.tracked_humans[idx] = {"tracker": tracker, "data": human}

        rospy.loginfo("exiting helper_rgb_human_tracker")

    def helper_find_angle_centroid(self):
        rospy.loginfo("inside helper_find_angle_centroid")
        try:
            for idx, tracked_human in list(self.tracked_humans.items()):
                x, y, w, h = tracked_human["data"]["bounding_box"]
                centroid_x_2d = x + w // 2
                centroid_y_2d = y + h // 2
                centroid_2d = [centroid_x_2d, centroid_y_2d]

                # Check if the centroid coordinates are within the depth image range
                if (
                    centroid_x_2d < self.full_depth_img.shape[1]
                    and centroid_y_2d < self.full_depth_img.shape[0]
                ):
                    depth_at_centroid = self.full_depth_img[
                        centroid_y_2d, centroid_x_2d
                    ]

                    # Calculate the 3D world coordinates of the centroid
                    X = (
                        (centroid_x_2d - self.depth_cx)
                        * depth_at_centroid
                        / self.depth_fx
                    )
                    Y = (
                        (centroid_y_2d - self.depth_cy)
                        * depth_at_centroid
                        / self.depth_fy
                    )
                    Z = depth_at_centroid

                    # Calculate the angles
                    body_angle_x = np.arctan2(X, Z)
                    body_angle_y = np.arctan2(Y, Z)

                    self.tracked_humans[idx]["data"]["body_angle_y"] = np.degrees(
                        body_angle_y
                    )
                    self.tracked_humans[idx]["data"]["body_angle_x"] = np.degrees(
                        body_angle_x
                    )
                    self.tracked_humans[idx]["data"]["centroid_2d"] = centroid_2d
                else:
                    self.tracked_humans[idx]["data"]["body_angle_y"] = np.degrees(0)
                    self.tracked_humans[idx]["data"]["body_angle_x"] = np.degrees(0)
                    self.tracked_humans[idx]["data"]["centroid_2d"] = [0, 0]
                    rospy.loginfo(
                        f"centroid outside depth data, x - {centroid_x_2d} and y is {centroid_y_2d} whereas shape of depth img is- {self.full_depth_img.shape}"
                    )
            rospy.loginfo("exiting helper_find_angle_centroid")

        except Exception as e:
            rospy.loginfo(traceback.format_exc())
            rospy.loginfo(f"Error in helper_find_angle_centroid: {e}")

    def helper_detect_social_interactions(self):
        social_interactions = []
        interacting_human_indices = set()
        list_of_humans = self.tracked_humans
        for i in range(len(list_of_humans)):
            for j in range(i + 1, len(list_of_humans)):
                human1, human2 = list_of_humans[i], list_of_humans[j]
                h1_centroid_2d = human1["data"]["centroid_2d"]
                h2_centroid_2d = human2["data"]["centroid_2d"]

                h1_centroid_3d = self.compute_3d_centroid(h1_centroid_2d)
                h2_centroid_3d = self.compute_3d_centroid(h2_centroid_2d)
                self.tracked_humans[i]["data"]["centroid_3d"] = h1_centroid_3d
                self.tracked_humans[j]["data"]["centroid_3d"] = h2_centroid_3d
                h1_angle_x = np.radians(human1["data"]["body_angle_x"])
                h1_angle_y = np.radians(human1["data"]["body_angle_y"])
                h2_angle_x = np.radians(human2["data"]["body_angle_x"])
                h2_angle_y = np.radians(human2["data"]["body_angle_y"])

                vector1 = [
                    np.sin(h1_angle_x) * np.cos(h1_angle_y),
                    np.sin(h1_angle_y),
                    np.cos(h1_angle_x) * np.cos(h1_angle_y),
                ]
                vector2 = [
                    np.sin(h2_angle_x) * np.cos(h2_angle_y),
                    np.sin(h2_angle_y),
                    np.cos(h2_angle_x) * np.cos(h2_angle_y),
                ]

                # Calculate the angle between the vectors
                angle_diff = np.degrees(
                    np.arccos(
                        np.dot(vector1, vector2)
                        / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                    )
                )

                # Compute the real-world distance between the centroids
                real_distance = np.linalg.norm(h1_centroid_3d - h2_centroid_3d) / 1000.0

                def add_interaction(name):
                    social_interactions.append(
                        f"{name} interaction between human {i} and human {j} #angle-{angle_diff}, #dist-{real_distance}"
                    )
                    interacting_human_indices.add(i)
                    interacting_human_indices.add(j)

                conditions = {
                    "N-shape": angle_diff < 30 and (0.5 < real_distance < 4.3),
                    "Vis-a-vis": angle_diff > 150 and (0.5 < real_distance < 4.3),
                    "V-shape": angle_diff < 45 and (0.5 < real_distance < 3.3),
                    "L-shape": (85 < angle_diff < 95) and (0.5 < real_distance < 4.3),
                    "C-shape": (130 < angle_diff < 140) and (0.5 < real_distance < 4.3),
                    "Side-by-side": angle_diff < 10 and (0.5 < real_distance < 3.8),
                }

                # This priority list is for testing purposes and can be edited for future use.
                priority = [
                    "N-shape",
                    "Vis-a-vis",
                    "V-shape",
                    "L-shape",
                    "C-shape",
                    "Side-by-side",
                ]

                for interaction_type in priority:
                    if conditions[interaction_type]:
                        add_interaction(interaction_type)
                        break

        return social_interactions, interacting_human_indices

    def compute_3d_centroid(self, centroid_2d):
        centroid_x, centroid_y = centroid_2d
        depth = self.full_depth_img[centroid_y, centroid_x]

        Z = depth
        X = (centroid_x - self.depth_cx) * Z / self.depth_fx
        Y = (centroid_y - self.depth_cy) * Z / self.depth_fy
        return np.array([X, Y, Z])

    def helper_update_display_images(
        self,
        img_name,
        img,
        interacting_human_indices,
        conversation_center_2d=None,
        conversation_center_3d=None,
    ):
        for idx, tracked_human in list(self.tracked_humans.items()):
            x, y, w, h = tracked_human["data"]["bounding_box"]
            # contours = tracked_human['data']['contours']
            human_number = str(idx)  # Get the human number (index)
            if idx in interacting_human_indices:
                cv2.rectangle(
                    img, (x, y), (x + w, y + h), (0, 255, 0), 2
                )  # Green color for interacting humans
            else:
                cv2.rectangle(
                    img, (x, y), (x + w, y + h), (0, 255, 255), 2
                )  # Yellow color for non-interacting humans

        #  This condition will let us draw the 2D conversation center if available
        if conversation_center_2d is not None:
            center_x_2d, center_y_2d = [int(coord) for coord in conversation_center_2d]
            cv2.circle(
                img, (center_x_2d, center_y_2d), 10, (255, 255, 255), -1
            )  # White circle

        # Similarly, we can display the 3D conversation center as text if available
        if conversation_center_3d is not None:
            text = f"3D Center: ({conversation_center_3d[0]:.2f}, {conversation_center_3d[1]:.2f}, {conversation_center_3d[2]:.2f})"
            # Black shadow before the main white text to improve text readability
            cv2.putText(
                img,
                text,
                (10, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            # Main text in white color
            cv2.putText(
                img,
                text,
                (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        rospy.loginfo("inside display_image")
        cv2.imshow(img_name, img)
        cv2.waitKey(1)
        rospy.loginfo("exiting display_image")

    def update_tracked_humans_with_detected(self):
        # Update tracked humans with new bounding boxes from detection
        for idx, human in enumerate(self.humans_data):
            if idx in self.tracked_humans:
                # Update bounding box and centroid
                self.tracked_humans[idx]["data"]["bounding_box"] = human["bounding_box"]
                self.tracked_humans[idx]["data"]["centroid_2d"] = human["centroid_2d"]

    def calculate_conversation_centroid_2D_3D(self, interacting_human_indices):
        """
        Calculate the 2D and 3D centroid of a conversation based on the tracked humans and their interactions.
        """
        rospy.loginfo("entered calculate_conversation_centroid_2D_3D")
        total_x_2d = 0
        total_y_2d = 0
        total_depth = 0
        total_humans = 0

        for idx in interacting_human_indices:
            human_data = self.tracked_humans[idx]["data"]
            x, y, w, h = human_data["bounding_box"]
            # Calculate the average x and y coordinates for the bounding box
            avg_x = x + w // 2

            avg_y = y + h // 2
            # Sum these coordinates for 2D centroid calculation
            total_x_2d += avg_x
            total_y_2d += avg_y
            # Calculate the average depth at the bounding box's centroid

            depth_at_centroid = self.full_depth_img[avg_y, avg_x]
            weighted_depth = self.calculate_weighted_depth(x, y, w, h)
            total_depth += weighted_depth

            total_humans += 1

        if total_humans == 0:
            return None, None

        # Calculate the 2D centroid of the conversation
        conversation_center_x_2d = total_x_2d / total_humans
        conversation_center_y_2d = total_y_2d / total_humans

        # Calculate the average depth at the 2D centroid of the conversation
        conversation_center_depth = total_depth / total_humans

        # Convert 2D centroid and depth to 3D position
        conversation_center_x_3d = (
            (conversation_center_x_2d - self.depth_cx)
            * conversation_center_depth
            / self.depth_fx
        )
        conversation_center_y_3d = (
            (conversation_center_y_2d - self.depth_cy)
            * conversation_center_depth
            / self.depth_fy
        )
        conversation_center_z_3d = conversation_center_depth

        rospy.loginfo("exiting calculate_conversation_centroid_2D_3D")
        return (conversation_center_x_2d, conversation_center_y_2d), (
            conversation_center_x_3d,
            conversation_center_y_3d,
            conversation_center_z_3d,
        )

    def calculate_weighted_depth(self, x, y, w, h):
        depth_values = self.full_depth_img[y : y + h, x : x + w]
        weighted_depth = np.mean(depth_values)
        return weighted_depth

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = ConversationDetectionNode()
    node.run()
