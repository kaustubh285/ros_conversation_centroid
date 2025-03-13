### **README.md**  

# **Conversation Detector**  

This script detects potential human conversations in a scene based on bounding box proximity, depth similarity, and facing direction. It groups participants into structured conversations based on these criteria.  

## **Logic Overview**  

1. **Extract Human Data**  
   - Each detected human has attributes like bounding box, confidence score, pose (2D & 3D), and depth values.  

2. **Pairwise Comparison**  
   - Each human is compared with every other human in the scene.  

3. **Bounding Box Proximity Check**  
   - If two humans are spatially close based on their bounding box centroids, they are considered for interaction.  

4. **Depth Similarity Check**  
   - Ensures that two humans are at a similar depth (distance from the camera) within a set threshold.  

5. **Facing Direction Analysis**  
   - Determines if the two humans are oriented towards each other, suggesting possible engagement in conversation.  

6. **Grouping Logic**  
   - If two humans pass the above checks, they are grouped into the same conversation.  
   - If a human is already in a group, they are added to it instead of creating a new group.  

7. **Structured Output**  
   - Conversations are stored in a dictionary format:  
     ```json
     {
       "conversations": [
         { "group": 1, "participants": [0, 1] },
         { "group": 2, "participants": [2, 3] }
       ]
     }
     ```  

## **Usage**  
- The function returns a structured list of conversations, identifying groups of interacting humans.  

This ensures an efficient and organized way to detect and represent human interactions in a given environment. 🚀