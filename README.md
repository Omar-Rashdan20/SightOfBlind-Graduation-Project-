# SightOfBlind

**SightOfBlind** is a real-time assistive system designed to support blind and visually impaired individuals by enabling object-based indoor navigation using voice commands and spoken feedback. It is fully offline, wearable, and powered by edge AI using a Raspberry Pi 5 and YOLOv11 object detection.

---

## 🎯 Project Overview

Visually impaired individuals face constant challenges in navigating unfamiliar environments. SightOfBlind aims to restore autonomy by allowing users to say commands like “Where is the door?” or “Find the chair,” and receive real-time verbal guidance based on AI-powered object recognition and direction awareness.

---

## ⚙️ Key Features

- 🎙️ Natural voice command interface via Speech-to-Text (STT)
- 📷 Real-time indoor object detection using YOLOv11n (TFLite)
- 🔊 Audio navigation feedback using Text-to-Speech (pyttsx3)
- ⚡ Runs completely offline on Raspberry Pi 5
- 🎒 Compact, wearable, and power-efficient design
- 💡 Directional logic: tells the user if the object is “in front,” “to your right,” etc how many steps away from you and the size of object.

---

## 🧠 Technologies Used

| Component           | Technology                     |
|--------------------|----------------------------------|
| Hardware           | Raspberry Pi 5, USB Webcam, AirPods |
| Object Detection   | YOLOv11n (custom & pre-trained) |
| Voice Recognition  | SpeechRecognition (Google STT) |
| Audio Feedback     |  pyttsx3 (Text-to-Speech)      |
| Programming        | Python 3                        |
| Vision Processing  | OpenCV                          |

---

## 🛠 Hardware Architecture

- **Raspberry Pi 5 (8GB)** – main processor
- **USB Webcam** – captures front-facing visual input
- **AirPods (Mic + Audio)** – for voice command and feedback
- **Cooling Fan + Protective Case** – ensures thermal stability
- **Portable Power Bank** – for mobile, wearable operation

---

## 📁 Project Structure

```bash
SightOfBlind/
├── models/               
├── docs/               
├── SightOfBlind.py       
├── requirements.txt      
├── media/                
└── README.md
```
---
## 🚀 Setup & Execution

1. Clone the Repository
```bash
git clone https://github.com/Obeidatt/SightOfBlind-Graduation-Project-.git
cd SightOfBlind
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run the Application
```bash
python3 SightOfBlind.py
```
- ✅ Make sure Bluetooth is connected and Pi Camera is enabled.
  
## 📊 Performance Highlights

- ✅ **mAP@0.5**: 89.3%
- ✅ **Precision**: 90%
- ✅ **Recall**: 86%
- ✅ **FPS**: 8–12 on Raspberry Pi 5 (TFLite, float16)
- ✅ **Command-to-Feedback Time**: ~2 sec

Tested in classrooms, hallways, and dorm environments with high voice command accuracy and object recognition robustness.

## 📚 Research Basis
This system is based on a graduation project submitted to:  
Yarmouk University  
Faculty of Computer Science and Information Technology  
Department of Information Systems  
Semester: Second 2024/2025  
Students: Omar Rashdan, Anas Obeidat, Momen Bani-Ali 
Supervisor: Dr. Enas Alikhashashneh  
> 📄 Full project report available in /docs.
## 🔒 License

This repository is protected under intellectual property rights.

**All rights reserved © 2025 by the authors.**  
No part of this codebase may be copied, modified, distributed, or used in other projects without **explicit written permission**.

Contact: [@Omar-Rashdan20](https://github.com/Omar-Rashdan20) or Email: rashdanomar15@gmail.com.

## 📬 Contact

For access requests, academic inquiries, or demo opportunities:

**Omar Mahmoud Rashdan**  
📧 Email: *rashdanomar15@gmail.com*  
🔗 GitHub: [https://github.com/Omar-Rashdan20](https://github.com/Obeidatt)

