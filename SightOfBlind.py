import cv2
import speech_recognition as sr
import math
import time
from collections import defaultdict
from ultralytics import YOLO
import pyttsx3

class SightOfBlind:
    def __init__(self):
        # Initialize YOLO models
        self.detector = CombinedYOLODetector(custom_model="fibest.pt", yolo_model="yolov11n.pt")
        
        # Initialize voice assistant
        self.voice_assistant = VoiceAssistant()
        
        # Updated room mappings
        self.room_mappings = {
            "Kitchen": ["Sink", "Refrigerator", "Oven", "Microwave", "Coffee maker", "Toaster", "Kettle", 
                        "Bowl", "Bottle", "Cup", "Fork", "Knife", "Spoon", "Scissors", "Gas Stove", 
                        "Blender", "Dishwasher", "Trash can", "Clock", "Cutting board"],
            
            "Bedroom": ["Bed", "Wardrobe", "Lamp", "Desk", "Table lamp", "Clock", "Book", "Shelf", 
                        "Pillow", "Mirror", "Window", "Curtains", "Dresser", "Alarm clock", "Nightstand"],
            
            "Bathroom": ["Toilet", "Sink", "Mirror", "Shower", "Bathtub", "Soap dispenser", "Toothbrush", 
                        "Towel", "Toilet paper", "Hairdryer", "Scale", "Washing Machine", "Trash can"],
            
            "Living Room": ["Couch", "TV", "Table", "Chair", "Lamp", "Remote", "Book", "Vase", "Window", 
                            "Curtains", "Picture", "Clock", "Rug", "Shelf", "Plant", "Fireplace", "Coffee table"],
            
            "Office": ["Laptop", "Desk", "Chair", "Monitor", "Keyboard", "Mouse", "Printer", "Book", 
                        "Pen", "Paper", "Scissors", "Stapler", "Light Switch", "Trash can", "Calendar"],
            
            "Hallway": ["Door", "Rug", "Picture", "Light Switch", "Mirror", "Stairs", "Umbrella stand", 
                        "Fire Extinguisher", "Window", "Table lamp"],
            
            "Entrance": ["Door", "Keys", "Backpack", "Umbrella", "Shoes", "Coat", "Hat", "Light Switch", 
                        "Mirror", "Wallet", "Security camera"],
            
            "Utility Room": ["Washing Machine", "Dryer", "Iron", "Cleaning supplies", "Vacuum cleaner", 
                            "Water Cooler", "Fire Extinguisher", "Tools", "Trash can"]
        }
        
        # Updated primary room objects
        self.primary_room = {
            # Kitchen primary objects (4)
            "Refrigerator": "Kitchen",
            "Oven": "Kitchen", 
            "Microwave": "Kitchen",
            "Gas Stove": "Kitchen",
            
            # Living Room primary objects (4)
            "Couch": "Living Room",
            "TV": "Living Room",
            "Coffee table": "Living Room",
            "Remote": "Living Room",
            
            # Bathroom primary objects (4)
            "Toilet": "Bathroom",
            "Shower": "Bathroom",
            "Bathtub": "Bathroom",
            "Towel": "Bathroom",
            
            # Entrance primary objects (3)
            "Door": "Entrance",
            "Keys": "Entrance",
            "Shoes": "Entrance",
            
            # Hallway primary objects (3)
            "Stairs": "Hallway",
            "Light Switch": "Hallway",
            "Fire Extinguisher": "Hallway",
            
            # Utility Room primary objects (3)
            "Washing Machine": "Utility Room",
            "Dryer": "Utility Room",
            "Cleaning supplies": "Utility Room",
            
            # Bedroom primary objects (4)
            "Bed": "Bedroom",
            "Wardrobe": "Bedroom",
            "Pillow": "Bedroom",
            "Nightstand": "Bedroom",
            
            # Office primary objects (4)
            "Laptop": "Office",
            "Monitor": "Office",
            "Keyboard": "Office",
            "Printer": "Office"
        }
        
        # Build object to room mapping
        self.object_to_room = {}
        for room, objects in self.room_mappings.items():
            for obj in objects:
                self.object_to_room.setdefault(obj.lower(), []).append(room)
        
        # List of objects from the coco dataset and additional items
        self.object_categories = [
            "Person", "Chair", "Couch", "Bed", "Table", "TV", "Laptop", "Bottle", 
            "Cup", "Bowl", "Fork", "Knife", "Spoon", "Microwave", "Oven", "Sink", 
            "Refrigerator", "Book", "Clock", "Scissors", "Toilet", "Door", 
            "Remote", "Keyboard", "Mouse", "Cell phone", "Backpack", "Keys",
            "Mirror", "Shower", "Towel", "Washing Machine", "Lamp", "Pillow"
        ]
        
        # Combine all room objects with object categories 
        for objects in self.room_mappings.values():
            for obj in objects:
                if obj not in self.object_categories:
                    self.object_categories.append(obj)
        
        # Remove duplicates and sort alphabetically
        self.object_categories = sorted(list(set(self.object_categories)))
        
        # Last direction announcement time
        self.last_announcement_time = 0
        self.announcement_interval = 2  # seconds
    
    def get_compass_direction(self, bbox, img_width, img_height):
        """Calculate compass direction (NEWS) and distance for an object"""
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        frame_center_x, frame_center_y = img_width / 2, img_height / 2

        # Calculate relative position
        dx = x_center - frame_center_x
        dy = y_center - frame_center_y

        # Determine primary direction
        if abs(dx) > abs(dy):
            if dx > 0:
                primary_direction = "East"
            else:
                primary_direction = "West"
        else:
            if dy > 0:
                primary_direction = "South"
            else:
                primary_direction = "North"

        # Add secondary direction for diagonal cases
        direction = primary_direction
        if abs(dx) > 20 and abs(dy) > 20:  # Threshold for diagonal
            if dy > 0:
                if dx > 0:
                    direction = "Southeast"
                else:
                    direction = "Southwest"
            else:
                if dx > 0:
                    direction = "Northeast"
                else:
                    direction = "Northwest"

        # Calculate distance
        distance = math.hypot(dx, dy)
        distance_ratio = distance / math.hypot(frame_center_x, frame_center_y)

        if distance_ratio < 0.3:
            steps = "very close, about 1 step"
        elif distance_ratio < 0.5:
            steps = "about 2 to 3 steps"
        elif distance_ratio < 0.7:
            steps = "about 4 to 5 steps"
        else:
            steps = "more than 6 steps"

        return direction, steps
    
    def estimate_object_size(self, bbox, img_width, img_height):
        """Estimate the size of the object relative to the frame"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        relative_size = (width * height) / (img_width * img_height)

        if relative_size > 0.5:
            return "very large"
        elif relative_size > 0.25:
            return "large"
        elif relative_size > 0.1:
            return "medium-sized"
        elif relative_size > 0.02:
            return "small"
        else:
            return "very small"
    
    def determine_current_room(self, detected_objects, confidence_threshold=0.):
        """Determine which room the user is in based on detected objects"""
        room_scores = {room: 0 for room in self.room_mappings.keys()}
        for obj in detected_objects:
            label = obj['name'].lower()
            conf = obj['conf']
            if conf >= confidence_threshold and label in self.object_to_room:
                for room in self.object_to_room[label]:
                    weight = 2 if label in self.primary_room and self.primary_room[label] == room else 1
                    room_scores[room] += conf * weight

        best_room = max(room_scores, key=room_scores.get)
        return best_room if room_scores[best_room] > 0 else None
    
    def find_object(self, detected_objects, target_name):
        """Find a specific object in the list of detected objects"""
        for obj in detected_objects:
            if obj['name'].lower() == target_name.lower():
                return obj
        return None
    
    def guide_to_object(self, target_object, detected_objects, img_width, img_height):
        """Guide the user to the requested object"""
        current_time = time.time()
        if current_time - self.last_announcement_time < self.announcement_interval:
            return False
        
        self.last_announcement_time = current_time
        
        obj = self.find_object(detected_objects, target_object)
        if obj:
            direction, steps = self.get_compass_direction(obj['box'], img_width, img_height)
            size = self.estimate_object_size(obj['box'], img_width, img_height)
            self.voice_assistant.speak(f"Found it! The {obj['name']} is to the {direction}, {steps}. It looks {size}.")
            return True
        else:
            current_room = self.voice_assistant.most_common_room()
            target_rooms = self.object_to_room.get(target_object.lower(), [])

            if current_room and current_room not in target_rooms:
                door = self.find_object(detected_objects, "Door")
                if door:
                    direction, steps = self.get_compass_direction(door['box'], img_width, img_height)
                    self.voice_assistant.speak(f"The {target_object} is likely not here in the {current_room}. I see a door to the {direction}, {steps}. Let's move toward it.")
                else:
                    self.voice_assistant.speak(f"I don't find the {target_object} here. Try turning around slowly to find a door.")
            else:
                self.voice_assistant.speak(f"Scanning for {target_object}... Please move slowly.")
            return False
    
    def recognize_speech(self):
        """Listen for user speech command and convert to text"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                self.voice_assistant.speak("I didn't hear anything. Please try again.")
                return None
            except sr.UnknownValueError:
                self.voice_assistant.speak("Sorry, I could not understand what you said.")
                return None
            except sr.RequestError as e:
                self.voice_assistant.speak("Sorry, there was an error with the speech recognition service.")
                return None
    
    def run(self):
        """Main method to run the application"""
        self.voice_assistant.speak("Welcome to Sight of Blind. You can say 'find' followed by an object to find it, or 'where am I' to determine your location.")
        
        # Initialize the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.voice_assistant.speak("Error: Camera not accessible")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                # Listen for command
                self.voice_assistant.speak("What would you like to do?")
                command = self.recognize_speech()
                
                if not command:
                    continue
                
                # Process commands
                if "exit" in command or "quit" in command or "stop" in command:
                    self.voice_assistant.speak("Exiting Sight of Blind. Goodbye.")
                    break
                    
                elif "where am i" in command or "what room am i in" in command or "what's my location" in command or"where i" in command:
                    self.voice_assistant.speak("Determining your location...")
                    self.detect_room(cap)
                    
                elif "locate" in command or "find" in command or "take me to" in command:
                    # Extract target object from command
                    target_object = None
                    
                    for obj in self.object_categories:
                        if obj.lower() in command:
                            target_object = obj
                            break
                    
                    if target_object:
                        self.voice_assistant.speak(f"Searching for {target_object}. I'll guide you there.")
                        self.locate_object(cap, target_object)
                    else:
                        self.voice_assistant.speak("I didn't recognize what you're looking for. Please try again with a specific object.")
                
                else:
                    self.voice_assistant.speak("I didn't understand that command. You can say 'find' followed by an object, 'where am I', or 'exit'.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def detect_room(self, cap):
        """Detect which room the user is in"""
        # Take multiple readings to determine the room more accurately
        room_detection_count = 0
        max_detection_attempts = 10
        
        self.voice_assistant.speak("Please wait while I determine which room you are in...")
        
        while room_detection_count < max_detection_attempts:
            ret, frame = cap.read()
            if not ret:
                self.voice_assistant.speak("Camera error. Stopping.")
                return
                
            # Process frame
            _, detections = self.detector.detect(frame)
            
            # Determine current room
            current_room = self.determine_current_room(detections)
            self.voice_assistant.update_room_history(current_room)
            
            room_detection_count += 1
            time.sleep(0.5)  # Short pause between readings
        
        stable_room = self.voice_assistant.most_common_room()
        
        if stable_room:
            self.voice_assistant.speak(f"You appear to be in the {stable_room}.")
            
            # Also announce visible objects
            objects_seen = set()
            for obj in detections:
                objects_seen.add(obj['name'])
            
            if objects_seen:
                objects_list = ", ".join(list(objects_seen)[:5])  # Limit to 5 objects to avoid long speech
                self.voice_assistant.speak(f"I can see: {objects_list}")
        else:
            self.voice_assistant.speak("I couldn't determine which room you're in.")
    
    def locate_object(self, cap, target_object):
        """Locate a specific object and guide the user to it"""
        search_time_start = time.time()
        search_timeout = 30  # seconds
        object_found_count = 0
        required_confirmations = 2  # Number of consecutive detections needed to confirm object is found
        
        self.voice_assistant.speak(f"Looking for {target_object}. Please move the camera slowly.")
        
        while time.time() - search_time_start < search_timeout:
            ret, frame = cap.read()
            if not ret:
                self.voice_assistant.speak("Camera error. Stopping search.")
                return
                
            # Process frame
            _, detections = self.detector.detect(frame)
            img_height, img_width = frame.shape[:2]
            
            # Update room history based on what we see
            current_room = self.determine_current_room(detections)
            self.voice_assistant.update_room_history(current_room)
            
            # Check if target object is found
            obj = self.find_object(detections, target_object)
            if obj:
                object_found_count += 1
                
                # Guide to object
                self.guide_to_object(target_object, detections, img_width, img_height)
                
                # If object has been consistently detected, consider it found
                if object_found_count >= required_confirmations:
                    self.voice_assistant.speak(f"Successfully located the {target_object}")
                    return
                    
                time.sleep(0.5)  # Brief pause when object is detected
            else:
                object_found_count = 0  # Reset counter if object not found
                
                # Provide guidance when object is not found
                current_room = self.voice_assistant.most_common_room()
                target_rooms = self.object_to_room.get(target_object.lower(), [])

                current_time = time.time()
                if current_time - self.last_announcement_time >= self.announcement_interval:
                    self.last_announcement_time = current_time
                    
                    if current_room and current_room not in target_rooms:
                        door = self.find_object(detections, "Door")
                        if door:
                            direction, steps = self.get_compass_direction(door['box'], img_width, img_height)
                            self.voice_assistant.speak(f"The {target_object} is likely not here in the {current_room}. I see a door to the {direction}, {steps}. Let's move toward it.")
                        else:
                            self.voice_assistant.speak(f"I don't find the {target_object} here. Try turning around slowly to find a door.")
                    else:
                        self.voice_assistant.speak(f"Still scanning for {target_object}... Please move slowly.")
        
        # If we reach here, the search timed out
        self.voice_assistant.speak(f"I couldn't find the {target_object} after searching ")
        return


class VoiceAssistant:
    def __init__(self):
        # Initialize pyttsx3 instead of gTTS
        self.engine = pyttsx3.init()
        self.room_history = []
        self.last_spoken = ""
        
        # Configure pyttsx3 settings
        self.engine.setProperty('rate', 150)    # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    
    def speak(self, text):
        # Don't repeat the same message
        if text == self.last_spoken:
            return
            
        self.last_spoken = text
        print(f"Assistant: {text}")
        
        # Use pyttsx3 to speak
        self.engine.say(text)
        self.engine.runAndWait()
    
    def update_room_history(self, room):
        if room:  # Only update if room is not None
            self.room_history.append(room)
            if len(self.room_history) > 5:
                self.room_history.pop(0)
    
    def most_common_room(self):
        if not self.room_history:
            return None
        return max(set(self.room_history), key=self.room_history.count)


class CombinedYOLODetector:
    def __init__(self, custom_model=r"C:\Users\96279\OneDrive\Desktop\Project_Final\fibest_float16.tflite", yolo_model=r"C:\Users\96279\OneDrive\Desktop\Project_Final\yolov11n_float16.tflite", conf_thresh=0.3, iou_thresh=0.5):
        self.custom_model = YOLO(custom_model)
        self.yolo_model = YOLO(yolo_model)
        self.custom_classes = {k + 100: v for k, v in self.custom_model.names.items()}
        self.yolo_classes = self.yolo_model.names
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        custom_results = self.custom_model.predict(img_rgb, verbose=False, conf=self.conf_thresh)[0]
        yolo_results = self.yolo_model.predict(img_rgb, verbose=False, conf=self.conf_thresh)[0]
        detections = self._process_results(custom_results, yolo_results)
        return self._draw_boxes(detections, frame), detections

    def _process_results(self, custom_results, yolo_results):
        boxes = []
        for box in custom_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append({
                'box': (x1, y1, x2, y2),
                'conf': float(box.conf[0]),
                'name': self.custom_classes[int(box.cls[0]) + 1000]
            })
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append({
                'box': (x1, y1, x2, y2),
                'conf': float(box.conf[0]),
                'name': self.yolo_classes[int(box.cls[0])]
            })
        return self._nms(boxes)

    def _nms(self, boxes):
        if not boxes:
            return []
        boxes.sort(key=lambda x: x['conf'], reverse=True)
        result = []
        while boxes:
            current = boxes.pop(0)
            result.append(current)
            boxes = [b for b in boxes if self._iou(current['box'], b['box']) < self.iou_thresh or current['name'] != b['name']]
        return result

    def _iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        x_left, y_top = max(x1_min, x2_min), max(y1_min, y2_min)
        x_right, y_bottom = min(x1_max, x2_max), min(y1_max, y2_max)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        return intersection / (box1_area + box2_area - intersection)

    def _draw_boxes(self, detections, frame):
        img = frame.copy()
        counts = defaultdict(int)
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['name']} {det['conf']:.2f}"
            counts[det['name']] += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos = 30
        for name, count in counts.items():
            cv2.putText(img, f"{name}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
        return img


def main():
    """Main function to run the application"""
    app = SightOfBlind()
    app.run()


if __name__ == "__main__":
    main()
