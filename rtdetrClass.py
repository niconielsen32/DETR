import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv

from ultralytics.vit import RTDETR


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = RTDETR("rtdetr-l.pt")
        
        self.CLASS_NAMES_DICT = self.model.model.names
        
        print(self.CLASS_NAMES_DICT)
    
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

    
    def plot_bboxes(self, results, frame):
        
        # Extract detections
        
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        
    
        class_id = class_id.astype(np.int32)
    
        
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=class_id,
                    )
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for xyxy, mask, confidence, class_id, track_id in detections]
    
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
           
            
            results = self.model.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index=0)
detector()