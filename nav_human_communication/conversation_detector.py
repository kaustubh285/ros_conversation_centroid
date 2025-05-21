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
import random
from msgs_nav_conversation.msg import (
    GroupMessage,
    CentroidInfo,
    Conversation,
    AgentInfo,
    Pose3D,
    Point2D,
    Point3D,
)
import torch

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
        self.all_social_interactions = None
        self.group_colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(10)]
        self.rgb_image_raw = self.create_subscription(Image, "/intel_realsense_r200_depth/image_raw", self.rgb_image_callback,10)
        self.depth_image_raw = self.create_subscription(Image, "/intel_realsense_r200_depth/depth/image_raw",self.depth_image_callback,10)
        self.flagged_humans = set()  
        # self.group_publisher = self.create_publisher(json, "/group_data", 10)
        self.group_publisher = self.create_publisher(GroupMessage, "/group_data", 10)
    
    def rgb_image_callback(self, img_raw: Image):
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
        new_social_interaction = self.detect_conversation(self.tracked_humans)
        self.get_logger().info("Printing social interaction")
        self.get_logger().info(str(self.all_social_interactions))
            # Calculate centroids of conversation groups
        self.all_social_interactions = self.calculate_conversation_centroids(self.all_social_interactions, new_social_interaction, self.tracked_humans)
        self.get_logger().info("Conversation group centroids:")
        self.get_logger().info(str(new_social_interaction))
        # self.convert_to_group_msg(new_social_interaction)  

        msg = self.convert_to_group_message(new_social_interaction, self.tracked_humans)
        self.group_publisher.publish(msg)

        self.draw_conversation_circle(rgb_img, new_social_interaction, self.tracked_humans)
        
        for group in new_social_interaction.get("conversations", []):
            group_id = group["group"]
            centroid = group.get("centroid")
            if centroid:
                cv2.circle(rgb_img, (int(centroid[0]), int(centroid[1])), 10, (0, 255, 0), -1)
                cv2.putText(rgb_img, f"Group {group_id}", (int(centroid[0]), int(centroid[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        
        annotated_image = self.bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
        annotated_image.header.stamp = img_raw.header.stamp
        annotated_image.header.frame_id = img_raw.header.frame_id
        # self.image_publisher.publish(annotated_image)
        self.get_logger().info("Annotated image published")
        cv2.imshow("Annotated Image", rgb_img)
        cv2.waitKey(1)

    def convert_to_group_message(self, new_social_interaction, tracked_humans):
        group_msg = GroupMessage()
        centroid_info = CentroidInfo()
        centroid_info.conversations = []
        agent_info_list = []

        for group in new_social_interaction.get("conversations", []):
            conv = Conversation()
            conv.group = group["group"]
            conv.participants = group["participants"]
            if "centroid" in group and group["centroid"]:
                conv.centroid = list(map(float, group["centroid"]))
            centroid_info.conversations.append(conv)

        for agent_id, agent in tracked_humans.items():
            agent_msg = AgentInfo()
            agent_msg.id = int(agent_id)
            agent_msg.bbox = [int(x) for x in agent.get("bbox", [0,0,0,0])]
            agent_msg.confidence = float(agent.get("confidence", 0.0))
            # 2D pose
            # agent_msg.pose_2d = [Point2D(x=float(x), y=float(y)) for x, y in agent.get("2d_pose", [])]
            # 3D pose (if you want to fill Pose3D array)
            # agent_msg.pose = [Pose3D(x=float(x), y=float(y), confidence=float(conf)) for x, y, conf in agent.get("pose", [])]
            # 3D points
            for key in ["head_3d", "body_3d", "legs_3d"]:
                if key == "head_3d":
                    val = agent.get("3d_head")
                elif key == "body_3d":
                    val = agent.get("3d_body")
                elif key == "legs_3d":
                    val = agent.get("3d_legs")
                if val:
                    setattr(agent_msg, key, Point3D(x=float(val[0]), y=float(val[1]), z=float(val[2])))
            agent_info_list.append(agent_msg)

        group_msg.centroid_info = centroid_info
        group_msg.agents_info = agent_info_list
        return group_msg

    def draw_conversation_circle(self,rgb_img, centroids, tracked_humans):
        for group in centroids.get("conversations", []):
            group_id = group["group"]
            centroid = group.get("centroid")
            participants = group["participants"]

            if centroid:
                # Calculate the radius as the farthest human from the centroid
                max_distance = 0
                for participant_id in participants:
                    human = tracked_humans.get(participant_id, {})
                    for key in ["3d_body", "3d_legs", "3d_head"]:
                        position = human.get(key)
                        if position and not np.isinf(position[2]):  # Ensure valid depth
                            distance = np.linalg.norm(np.array(position[:2]) - np.array(centroid[:2]))
                            max_distance = max(max_distance, distance)

                # Draw the circle on the image
                # cv2.ellipse(rgb_img, (int(centroid[0]), int(centroid[1])), int(max_distance), (255, 0, 0), 2)
                cv2.ellipse(
                    rgb_img,
                    center=(int(centroid[0]), int(centroid[1])),
                    axes=(int(max_distance), int(0.35 * max_distance)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=self.group_colors[group_id % len(self.group_colors)],
                    thickness=2,
                    lineType=cv2.LINE_4,
                )

    def calculate_conversation_centroids(self, all_conversation, conversations, tracked_humans):

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

            
            if positions:
                new_centroid = np.mean(positions, axis=0)
                if all_conversation:
                    
                    historical_centroid = next(
                        (conv.get("centroid") for conv in all_conversation.get("conversations", []) if conv["group"] == group_id),
                        None
                    )
                    if historical_centroid:
                        # Weighted average: 65% new data, 35% historical data
                        new_centroid = 0.65 * np.array(new_centroid) + 0.35 * np.array(historical_centroid)
                group['centroid'] = tuple(new_centroid)

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

            # Ensure tensors are moved to CPU before converting to NumPy
            boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            keypoints = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else keypoints

            for box, kp in zip(boxes, keypoints):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
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

        return tracked_humans
   
    def detect_conversation(self, humans, depth_threshold=1.0, bbox_distance_threshold=100):
        conversations = {"conversations": []}
        human_ids = list(humans.keys())

        if len(human_ids) < 2:
            return conversations

        groups = []
        assigned = set()

        def bounding_boxes_close(bbox1, bbox2):
            
            x1_center, y1_center = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
            x2_center, y2_center = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2
            return np.linalg.norm([x1_center - x2_center, y1_center - y2_center]) < bbox_distance_threshold

        def depth_similar(human1, human2):
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

        if self.all_social_interactions:
            for group in self.all_social_interactions.get("conversations", []):
                for participant in group["participants"]:
                    if participant not in human_ids:
                        self.flagged_humans.add(participant)

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
                        for group in groups:
                            if human_ids[i] in group["participants"] or human_ids[j] in group["participants"]:
                                group["participants"].update([human_ids[i], human_ids[j]])
                                group_found = True
                                break
                        
                        if not group_found:
                            groups.append({"group": len(groups) + 1, "participants": {human_ids[i], human_ids[j]}})
                        
                        assigned.update([human_ids[i], human_ids[j]])

        
        if self.flagged_humans:
            for flagged_human in list(self.flagged_humans):
                for human_id in human_ids:
                    if flagged_human != human_id:
                        human1 = humans.get(flagged_human, {})
                        human2 = humans[human_id]

                        bbox1 = human1.get("bbox", (0, 0, 0, 0))
                        bbox2 = human2.get("bbox", (0, 0, 0, 0))

                        if bounding_boxes_close(bbox1, bbox2) and depth_similar(human1, human2):
                            conversation_type = self.are_facing_each_other(human1, human2)

                            if conversation_type != "No interaction":
                                group_found = False

                                for group in groups:
                                    if flagged_human in group["participants"] or human_id in group["participants"]:
                                        group["participants"].update([flagged_human, human_id])
                                        group_found = True
                                        break
                                
                                if not group_found:
                                    groups.append({"group": len(groups) + 1, "participants": {flagged_human, human_id}})
                                
                                assigned.update([flagged_human, human_id])
                                self.flagged_humans.remove(flagged_human)
                                break


        for group in groups:
            group["participants"] = list(group["participants"])
            conversations["conversations"].append(group)

        return conversations


    def calculate_facing_direction(self, human):
        keypoints = human.get('pose', [])
        if not keypoints:
            return None

        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6

        if len(keypoints) <= max(NOSE, LEFT_SHOULDER, RIGHT_SHOULDER):
            return None

        nose_2d = keypoints[NOSE][:2]
        left_shoulder_2d = keypoints[LEFT_SHOULDER][:2]
        right_shoulder_2d = keypoints[RIGHT_SHOULDER][:2]

        nose_depth = human.get("3d_head", (0, 0, None))[2]
        if nose_depth is None or np.isinf(nose_depth):
            nose_depth = human.get("3d_body", (0, 0, None))[2]

        left_shoulder_depth = human.get("3d_body", (0, 0, None))[2]
        right_shoulder_depth = human.get("3d_body", (0, 0, None))[2]

        if (nose_depth is None or left_shoulder_depth is None or 
            np.isinf(nose_depth) or np.isinf(left_shoulder_depth)):
            return None

        nose_3d = np.array([nose_2d[0], nose_2d[1], nose_depth])
        shoulder_center = np.array([
            (left_shoulder_2d[0] + right_shoulder_2d[0]) / 2,
            (left_shoulder_2d[1] + right_shoulder_2d[1]) / 2,
            (left_shoulder_depth + right_shoulder_depth) / 2
        ])

        facing_vector = nose_3d - shoulder_center
        norm = np.linalg.norm(facing_vector)
        return facing_vector / norm if norm > 0 else None

    

    def are_facing_each_other(self, human1, human2):

        def compute_2d_distance(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))
        
        def get_best_point(human):
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
        
        # thresholds 
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
        
        self.get_logger().info(f"[DEBUG] angle_diff: {angle_diff}")
        self.get_logger().info(f"[DEBUG] head_distance: {head_distance}")
        self.get_logger().info(f"[DEBUG] shape scores: {shape_scores}")
        
        best_shape = max(shape_scores, key=shape_scores.get)
        return best_shape if shape_scores[best_shape] > 0.5 else "No interaction"

def main(args=None):
    rclpy.init(args=args)

    node = ConversationDetector()
    rclpy.spin(node)

    rclpy.shutdown()


if __name__=="__main__":
    main()