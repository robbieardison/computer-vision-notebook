import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from typing import Tuple, Optional

# Load pre-trained DeepLabV3 model
def load_model():
    """Load and return a pre-trained DeepLabV3 model."""
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input frame
def preprocess_frame(frame: cv2.Mat) -> torch.Tensor:
    """Preprocess the frame for model inference."""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(frame).unsqueeze(0)

# Generate a segmentation mask
def generate_mask(model: torch.nn.Module, frame: cv2.Mat) -> cv2.Mat:
    """Generate a segmentation mask for the frame."""
    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)
    mask = output_predictions.byte().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = mask.astype('uint8')  # Ensure mask is in CV_8U format
        # Binarize the mask (without this, the foreground will be semi-transparent)
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    return mask

# Composite foreground and new background
def composite_frame(frame: cv2.Mat, mask: cv2.Mat, new_background: cv2.Mat) -> cv2.Mat:
    """Composite the foreground from the frame onto the new background."""
    # Ensure the new background is the same size as the frame
    new_background = cv2.resize(new_background, (frame.shape[1], frame.shape[0]))

    # Extract the foreground using the mask
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Extract the background from the new background
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(new_background, new_background, mask=background_mask)

    # Combine the foreground and new background
    result = cv2.add(foreground, background)
    return result

# Process a single frame
def process_frame(frame: cv2.Mat, model: torch.nn.Module, new_background: cv2.Mat) -> cv2.Mat:
    """Process a single frame: remove background and add new background."""
    mask = generate_mask(model, frame)
    return composite_frame(frame, mask, new_background)

# Process the entire video
def process_video(input_video_path: str, output_video_path: str, new_background_path: str) -> None:
    """Process the video: remove background and add new background."""
    # Load the new background image
    new_background = cv2.imread(new_background_path)
    if new_background is None:
        raise ValueError("Could not load background image.")

    # Load the model
    model = load_model()

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        result = process_frame(frame, model, new_background)

        # Write the frame to the output video
        out.write(result)

        # Display the frame (optional)
        cv2.imshow('Video with New Background', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "assets/background_test_set_1.mp4"  # Replace with your input video path
    output_video = "assets/background_removed_1.mp4"  # Replace with your output video path
    new_background = "assets/moon-retro-vector-art-7op8fanjrtajdzp0.jpg"  # Replace with your new background image path
    try:
        process_video(input_video, output_video, new_background)
    except Exception as e:
        print(f"Error: {e}")