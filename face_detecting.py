import mediapipe as mp
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import os
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pyautogui
from PIL import ImageGrab 

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for image_name in os.listdir(cls_dir):
                image_path = os.path.join(cls_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(self.class_to_idx[cls_name])

        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label



# Define MobileNetV3 model
model = mobilenet_v3_small(pretrained=False)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_ftrs, 16)  # Assuming num_classes is your number of classes
model.load_state_dict(torch.load('trained_model.pth'))  # Load your trained MobileNetV3 model
model.eval()

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils 
face_detection = mp_face_detection.FaceDetection()

# Define transformations for preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match MobileNetV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

dataset = CustomDataset(data_dir='./dataset_bitirme', transform=preprocess)

# Assuming you have access to your dataset classes, e.g., dataset.classes
class_names = dataset.classes

# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    # Capture screen frame
    screen = pyautogui.screenshot()
    screen = np.array(screen)
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)


    # Detect faces
    results = face_detection.process(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = screen.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Extract face region
            x, y, w, h = bbox
            face_region = screen[y:y+h, x:x+w]
            
            # Preprocess the detected face image
            preprocessed_face = preprocess(face_region)
            
            # Perform face recognition using your MobileNetV3 model
            with torch.no_grad():
                output = model(preprocessed_face.unsqueeze(0))
                # Process output to get recognition results
                # Get the class name from class_names using the predicted label
                predicted_label = torch.argmax(output).item()
                class_name = class_names[predicted_label]
            
                # Draw bounding box and recognition results with adjusted colors
                cv2.rectangle(screen, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue bounding box
                cv2.putText(screen, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Green text

    cv2.imshow('Face Recognition', screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()