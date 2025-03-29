#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import json


class ConversationDetector(Node):
    
    def __init__(self):
        super().__init__("conversation_detector_node")
        self.get_logger().info("Hello! Node has started")

        self.humans_data = []  
        self.tracked_humans = {}
        self.depth_img = None
        self.pose_model = YOLO('yolov8n-pose.pt')

        self.tracker = sv.ByteTrack()
        self.bridge = CvBridge()
        self.rgb_image_raw = self.create_subscription(Image, "/intel_realsense_r200_depth/image_raw", self.rgb_image_callback,10)
        self.depth_image_raw = self.create_subscription(Image, "/intel_realsense_r200_depth/depth/image_raw",self.depth_image_callback,10)
        
    
    def rgb_image_callback(self,img_raw:Image):
        self.get_logger().info("got rgb raw image!!!!")
        rgb_img = self.bridge.imgmsg_to_cv2(img_raw,desired_encoding="bgr8")
        self.tracked_humans = self.helper_rgb_human_detect(rgb_img)

        for human_id, human_data in self.tracked_humans.items():
            try:
                positions = self.calculate_3d_positions(human_data, self.depth_img)
                self.tracked_humans[human_id].update(positions)
                self.get_logger().info(str(positions))
            except Exception as e:
                self.get_logger().info('Error for '+str(human_id)+'||'+str(e))
        self.get_logger().info(str(self.tracked_humans))

        social_interactions = self.detect_conversation(self.tracked_humans)
        self.get_logger().info("Printing social interaction")
        self.get_logger().info(str(social_interactions))
            # Calculate centroids of conversation groups
        centroids = self.calculate_conversation_centroids(social_interactions, self.tracked_humans)
        self.get_logger().info("Conversation group centroids:")
        self.get_logger().info(str(centroids))

        # Annotate the image with conversation data
        for group in centroids.get("conversations", []):
            group_id = group["group"]
            centroid = group.get("centroid")
            if centroid:
                cv2.circle(rgb_img, (int(centroid[0]), int(centroid[1])), 10, (0, 255, 0), -1)
                cv2.putText(rgb_img, f"Group {group_id}", (int(centroid[0]), int(centroid[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Publish the annotated image
        annotated_image = self.bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
        annotated_image.header.stamp = img_raw.header.stamp
        annotated_image.header.frame_id = img_raw.header.frame_id
        # self.image_publisher.publish(annotated_image)
        self.get_logger().info("Annotated image published")
        cv2.imshow("Annotated Image", rgb_img)
        cv2.waitKey(1)


    def calculate_conversation_centroids(self, conversations, tracked_humans):
        """
        Calculate the centroid of each conversation group based on the 3D positions of participants.
        
        Args:
            conversations (dict): The conversation groups with participant IDs.
            tracked_humans (dict): The tracked humans' data with 3D positions.

        Returns:
            dict: A dictionary with group IDs as keys and their centroids as values.
        """
        centroids = {}

        for group in conversations.get("conversations", []):
            group_id = group["group"]
            participants = group["participants"]

            # Collect valid 3D positions of participants
            positions = []
            for participant_id in participants:
                human = tracked_humans.get(participant_id, {})
                for key in ["3d_body", "3d_legs", "3d_head"]:
                    position = human.get(key)
                    if position and not np.isinf(position[2]):  # Ensure valid depth
                        positions.append(position)
                        break

            # Calculate the centroid if there are valid positions
            if positions:
                centroid = np.mean(positions, axis=0)
                group['centroid'] = tuple(centroid)

        return conversations
    
    def calculate_3d_positions(self, human, depth_image):
        positions = {
            "2d_pose": [],
            "3d_head": None,
            "3d_body": None,
            "3d_legs": None
        }
        
        keypoints = human.get('pose', [])
        if not keypoints:
            human['error'] = 'No keypoints provided'
            return None

        indices = {
            "head": [0, 1, 2, 3, 4],  # Nose, Eyes, Ears
            "shoulders": [5, 6],  # Left Shoulder, Right Shoulder
            "body": [5, 6, 11, 12],  # Shoulders + Hips
            "legs": [11, 12, 13, 14, 15, 16]  # Legs
        }

        def get_depth(x, y):
            if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                depth = depth_image[int(y), int(x)]
                if np.isinf(depth) or depth <= 0:
                    window = depth_image[max(0, y-1):min(depth_image.shape[0], y+2),
                                        max(0, x-1):min(depth_image.shape[1], x+2)]
                    valid_depths = window[window > 0]
                    if valid_depths.size > 0:
                        return np.median(valid_depths)  
                    return None  
                return depth
            return None

        points = np.array([[kp[0], kp[1], kp[2] if len(kp) > 2 else 0.0] for kp in keypoints], dtype=float)

        positions['2d_pose'] = [(int(x), int(y)) for x, y, conf in points]
        def calculate_average_3d(indices_list):
            valid_points = []
            for i in indices_list:
                if i < len(points) and points[i, 2] > 0.2: 
                    x, y = int(points[i, 0]), int(points[i, 1])
                    z = get_depth(x, y)
                    if z is not None:
                        valid_points.append([x, y, z])
            return tuple(np.mean(valid_points, axis=0)) if valid_points else None

        
        positions['3d_head'] = calculate_average_3d(indices["head"] + indices["shoulders"])  
        positions['3d_body'] = calculate_average_3d(indices["body"])
        positions['3d_legs'] = calculate_average_3d(indices["legs"])

        avg_depth = np.median(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 2.0

        def fallback_3d(part):
            if positions[part] is None:
                positions[part] = (positions["3d_body"][0], positions["3d_body"][1], avg_depth * 0.5) if positions["3d_body"] else (0, 0, avg_depth * 0.5)

        fallback_3d("3d_head")
        fallback_3d("3d_body")
        fallback_3d("3d_legs")

        self.get_logger().info(str(positions))
        return positions

    def depth_image_callback(self,img_raw:Image):
        # self.get_logger().info("got depth raw image")
        depth_img = self.bridge.imgmsg_to_cv2(img_raw,desired_encoding="passthrough")
        self.depth_img = depth_img
        self.get_logger().info("got depth img")

        # cv2.imshow("depth_img",depth_img)
        # cv2.waitKey(1)

    def helper_rgb_human_detect(self, frame):

        def draw_skeleton(frame, keypoints):
            skeleton = [[6, 8], [8, 10], [5, 7], [7, 9], 
                        [6, 12], [5, 11], [12, 14], [11, 13], [14, 16], [13, 15]]
            for connection in skeleton:
                try:
                    if (keypoints[connection[0]][2] > 0.3 and 
                        keypoints[connection[1]][2] > 0.3):  
                        pt1 = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
                        pt2 = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                except IndexError:
                    self.get_logger().warn("Invalid keypoints detected.")
                    continue

        
        results = self.pose_model.predict(frame, conf=0.5, verbose=False)
        
        display_frame = frame.copy()
        tracked_humans = {}

        for result in results:
            
            boxes = result.boxes
            keypoints = result.keypoints

            for box, kp in zip(boxes, keypoints):
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                
                label = f"Human: {conf:.2f}"
                label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 10)
                cv2.putText(display_frame, label, label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                
                filtered_keypoints = []
                for x, y, k_conf in kp.data[0]:
                    if k_conf > 0.2:  
                        x, y = int(x), int(y)
                        
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            filtered_keypoints.append((x, y, k_conf))
                            cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)

                if len(filtered_keypoints) > 1:
                    draw_skeleton(display_frame, filtered_keypoints)

                tracked_humans[len(tracked_humans)] = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'pose': filtered_keypoints
                }

        # cv2.imshow("Human Detection with Poses", display_frame)
        # cv2.waitKey(1)

        return tracked_humans

   
    def detect_conversation(self, humans, depth_threshold=1.0, bbox_distance_threshold=100):
        """
        Detects potential human conversations and groups participants based on bounding box proximity,
        depth similarity, and facing direction.
        """
        conversations = {"conversations": []}
        human_ids = list(humans.keys())

        if len(human_ids) < 2:
            return conversations

        groups = []
        assigned = set()

        def bounding_boxes_close(bbox1, bbox2):
            """Check if bounding box centroids are within a given threshold distance."""
            x1_center, y1_center = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
            x2_center, y2_center = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2
            return np.linalg.norm([x1_center - x2_center, y1_center - y2_center]) < bbox_distance_threshold

        def depth_similar(human1, human2):
            """Check if the depth values of two humans are within a threshold."""
            depths1 = [human1.get("3d_head", (0, 0, np.inf))[2], 
                    human1.get("3d_body", (0, 0, np.inf))[2], 
                    human1.get("3d_legs", (0, 0, np.inf))[2]]
            depths2 = [human2.get("3d_head", (0, 0, np.inf))[2], 
                    human2.get("3d_body", (0, 0, np.inf))[2], 
                    human2.get("3d_legs", (0, 0, np.inf))[2]]

            valid_depths1 = [d for d in depths1 if not np.isinf(d)]
            valid_depths2 = [d for d in depths2 if not np.isinf(d)]

            if not valid_depths1 or not valid_depths2:
                return False  

            return any(abs(d1 - d2) <= depth_threshold for d1 in valid_depths1 for d2 in valid_depths2)

        for i in range(len(human_ids)):
            for j in range(i + 1, len(human_ids)):
                human1 = humans[human_ids[i]]
                human2 = humans[human_ids[j]]

                bbox1 = human1.get("bbox", (0, 0, 0, 0))
                bbox2 = human2.get("bbox", (0, 0, 0, 0))

                if bounding_boxes_close(bbox1, bbox2) and depth_similar(human1, human2):
                    conversation_type = self.are_facing_each_other(human1, human2)

                    if conversation_type != "No interaction":
                        group_found = False

                        # Check if these humans are already in a group
                        for group in groups:
                            if human_ids[i] in group["participants"] or human_ids[j] in group["participants"]:
                                group["participants"].update([human_ids[i], human_ids[j]])
                                group_found = True
                                break
                        
                        # If they are not in an existing group, create a new group
                        if not group_found:
                            groups.append({"group": len(groups) + 1, "participants": {human_ids[i], human_ids[j]}})
                        
                        assigned.update([human_ids[i], human_ids[j]])

        # Convert sets to lists for JSON compatibility
        for group in groups:
            group["participants"] = list(group["participants"])
            conversations["conversations"].append(group)

        return conversations


    def calculate_facing_direction(self, human):
        """
        Calculate a normalized 3D facing vector using the nose and shoulders.
        If the nose (3d_head) depth is invalid (inf), it falls back to the body depth.
        """
        keypoints = human.get('pose', [])
        if not keypoints:
            return None

        # Define indices for keypoints (assumes YOLO-pose ordering)
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6

        if len(keypoints) <= max(NOSE, LEFT_SHOULDER, RIGHT_SHOULDER):
            return None

        # Extract 2D positions from the pose keypoints
        nose_2d = keypoints[NOSE][:2]
        left_shoulder_2d = keypoints[LEFT_SHOULDER][:2]
        right_shoulder_2d = keypoints[RIGHT_SHOULDER][:2]

        # Retrieve depth values
        nose_depth = human.get("3d_head", (0, 0, None))[2]
        # If nose depth is invalid, use body depth as backup
        if nose_depth is None or np.isinf(nose_depth):
            nose_depth = human.get("3d_body", (0, 0, None))[2]

        left_shoulder_depth = human.get("3d_body", (0, 0, None))[2]
        right_shoulder_depth = human.get("3d_body", (0, 0, None))[2]

        # If any critical value is missing, we cannot compute the vector
        if (nose_depth is None or left_shoulder_depth is None or 
            np.isinf(nose_depth) or np.isinf(left_shoulder_depth)):
            return None

        # Form 3D points (using body depth for shoulders)
        nose_3d = np.array([nose_2d[0], nose_2d[1], nose_depth])
        shoulder_center = np.array([
            (left_shoulder_2d[0] + right_shoulder_2d[0]) / 2,
            (left_shoulder_2d[1] + right_shoulder_2d[1]) / 2,
            (left_shoulder_depth + right_shoulder_depth) / 2
        ])

        # Compute facing vector: from shoulder center to nose
        facing_vector = nose_3d - shoulder_center
        norm = np.linalg.norm(facing_vector)
        return facing_vector / norm if norm > 0 else None

    

    def are_facing_each_other(self, human1, human2):
        """Determine interaction type based on facing direction and 2D head distance (in pixels)."""

        def compute_2d_distance(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))
        
        def get_best_point(human):
            """Selects the best available 3D point (prefer body > legs > head) to avoid inf depth issues."""
            for key in ["3d_body", "3d_legs", "3d_head"]:
                point = human.get(key, None)
                if point is not None and not np.isinf(point[2]):
                    return point
            return None
        
        point1 = get_best_point(human1)
        point2 = get_best_point(human2)
        
        if point1 is None or point2 is None:
            return "invalid data"
        
        facing1 = self.calculate_facing_direction(human1)
        facing2 = self.calculate_facing_direction(human2)
        
        if facing1 is None or facing2 is None:
            return "invalid data"
        
        head_distance = compute_2d_distance(point1[:2], point2[:2])
        
        dot_product = np.dot(facing1, facing2)
        angle_diff = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        # Update ideal thresholds to pixel units
        # (These values are examples—adjust based on your camera and scene)
        shape_definitions = {
            "N-shape": (30, 50),        # ideal angle 30°, ideal head distance 50 px
            "Vis-a-vis": (150, 50),      # ideal angle 150°, ideal head distance 50 px
            "V-shape": (45, 40),         # ideal angle 45°, ideal head distance 40 px
            "L-shape": (90, 50),         # ideal angle 90°, ideal head distance 50 px
            "C-shape": (135, 50),        # ideal angle 135°, ideal head distance 50 px
            "Side-by-side": (10, 40),    # ideal angle 10°, ideal head distance 40 px
        }
        
        shape_scores = {}
        
        for shape, (ideal_angle, ideal_distance) in shape_definitions.items():
            angle_score = 1 - (abs(ideal_angle - angle_diff) / ideal_angle)
            distance_score = 1 - (abs(ideal_distance - head_distance) / ideal_distance)
            shape_scores[shape] = angle_score * 0.6 + distance_score * 0.4
        
        # Debug prints (remove in production)
        print("[DEBUG] angle_diff:", angle_diff)
        print("[DEBUG] head_distance:", head_distance)
        print("[DEBUG] shape scores:", shape_scores)
        
        best_shape = max(shape_scores, key=shape_scores.get)
        return best_shape if shape_scores[best_shape] > 0.5 else "No interaction"

def main(args=None):
    rclpy.init(args=args)

    node = ConversationDetector()
    rclpy.spin(node)

    rclpy.shutdown()


if __name__=="__main__":
    main()