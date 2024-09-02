import cv2
import time
import datetime
import winsound
import face_recognition
import numpy as np
import pygame  # Import pygame for audio playback

# Initialize pygame for audio playback
pygame.mixer.init()

# Load the alert sound
alert_sound = pygame.mixer.Sound("alert.mp3")  # Ensure the path is correct

# Initialize video capture
cap = cv2.VideoCapture(1)

# Load the Haar cascades for face and body detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Load known face images and create encodings
known_face_encodings = []
known_face_names = []

# Example: Load a known face
# Replace 'Nirosha_image.jpg' with the path to your image
image_of_person = face_recognition.load_image_file("Nirosha_image.jpg")
image_of_person_encoding = face_recognition.face_encodings(image_of_person)[0]

# Append the encoding and name
known_face_encodings.append(image_of_person_encoding)
known_face_names.append("Nirosha")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for face_recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize variables for face recognition
    face_names = []
    unrecognized_face_detected = False

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unauthorized_Person"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            unrecognized_face_detected = True  # Mark that an unrecognized face is detected

        face_names.append(name)

    # Draw rectangles around detected faces and label them
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Set color based on the name
        if name == "Nirosha":
            color = (255, 0, 0)  # Blue color for Nirosha
        else:
            color = (0, 0, 255)  # Red color for Unauthorized_Person

        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, color, 2)

    # Start recording only if an unrecognized face is detected
    if unrecognized_face_detected:
        if detection:
            timer_started = False  # Reset timer if already detecting
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started Recording!")
            winsound.Beep(1000, 1000)  # Alert sound for starting recording
            
            # Play the alert sound
            alert_sound.play()  # Play the alert sound when an unrecognized face is detected
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stopped Recording!')
                winsound.Beep(500, 1000)  # Alert sound for stopping recording
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)
        cv2.putText(frame, "Recording...", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Not Recording", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()