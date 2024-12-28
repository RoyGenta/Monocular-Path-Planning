import cv2
import torch
import numpy as np

# Load MiDaS model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
model.eval()

# Use CPU for inference
device = torch.device("cpu")
model.to(device)

# Load transformation pipeline
transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

# Initialize webcam
cap = cv2.VideoCapture(0)

def plan_path(depth_map):
    """
    Basic path planning algorithm.
    Divides depth map into three regions: left, center, and right,
    and determines the direction with the greatest average depth (farthest path).
    """
    height, width = depth_map.shape
    left = depth_map[:, :width // 3]
    center = depth_map[:, width // 3: 2 * width // 3]
    right = depth_map[:, 2 * width // 3:]

    # Calculate average depth for each region
    avg_left = np.mean(left)
    avg_center = np.mean(center)
    avg_right = np.mean(right)

    # Determine direction
    if avg_center >= avg_left and avg_center >= avg_right:
        direction = "FORWARD"
    elif avg_left >= avg_center and avg_left >= avg_right:
        direction = "LEFT"
    else:
        direction = "RIGHT"

    return direction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Convert frame to RGB
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform input frame to tensor and ensure correct dimensions
    input_tensor = transform(input_frame).to(device)  # Apply transformation
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Pastikan dimensi tensor adalah [1, 3, height, width]
    if input_tensor.dim() == 5:  # Jika ada dimensi ekstra
        input_tensor = input_tensor.squeeze(0)  # Hilangkan dimensi tambahan



    # Predict depth
    with torch.no_grad():
        depth = model(input_tensor)
        depth = depth.squeeze().cpu().numpy()

    # Normalize depth map to 0-255 and invert the scale
    normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    inverted_depth = 255 - normalized_depth  # Invert scale: closer = white, farther = black
    inverted_depth = inverted_depth.astype(np.uint8)

    # Perform path planning
    direction = plan_path(inverted_depth)

    # Overlay direction on the original frame
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Direction: {direction}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the original frame, inverted depth map, and path planning result
    cv2.imshow("Original Frame", display_frame)
    cv2.imshow("Inverted Depth Map", inverted_depth)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
