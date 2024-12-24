import cv2
import face_recognition
import numpy as np
import os
from datetime import timedelta

images = []
imageName = []
path = "TestImages"  # Folder where reference images are stored

# Load images from the folder
imagesList = os.listdir(path)
for name in imagesList:
    currentImage = cv2.imread(f'{path}/{name}')
    images.append(currentImage)
    imageName.append(os.path.splitext(name)[0])

print("Loaded image names:", imageName)

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if encoding exists
            encodeList.append(encodings[0])
        else:
            print(f"No face detected in image: {img}")
    return encodeList

# Encode all images
encodedList = findEncodings(images)
print("Encoding Complete")

# Initialize video capture
video_path = 'Videos/Low Light.mp4'  # Path to the video file
capture = cv2.VideoCapture(video_path)

# Check if video is opened successfully
if not capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video frame rate for timestamp calculation
fps = capture.get(cv2.CAP_PROP_FPS)

frame_count = 0  # Count frames for naming saved images
frame_interval = 5  # Process every 5th frame
threshold = 0.5  # Set threshold for face match (lower is stricter)

while True:
    success, image = capture.read()

    # Check if a frame is successfully captured
    if not success:
        print("End of video or error in frame capture.")
        break

    print(f"Processing frame {frame_count}")  # Debugging frame processing

    if frame_count % frame_interval == 0:
        # Calculate timestamp
        timestamp = timedelta(seconds=frame_count / fps)

        # Convert frame to RGB for face recognition
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces and encode them
        facesCurFrame = face_recognition.face_locations(imageRGB)
        encodesCurFrame = face_recognition.face_encodings(imageRGB, facesCurFrame)

        print(f"Detected {len(facesCurFrame)} faces in frame {frame_count}")  # Debugging face detection

        # Match each detected face
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodedList, encodeFace)
            faceDis = face_recognition.face_distance(encodedList, encodeFace)

            # Apply threshold to face match
            if min(faceDis) < threshold:
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = imageName[matchIndex].upper()
                    print(f"Match found: {name} at {timestamp}")

                    # Get face coordinates
                    top, right, bottom, left = faceLoc

                    # Draw rectangle around face
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(image, (left, bottom), (right, bottom + 35), (0, 255, 0), cv2.FILLED)

                    # Put the name text below the rectangle
                    font = cv2.FONT_HERSHEY_COMPLEX
                    fontScale = 0.7
                    fontThickness = 2
                    cv2.putText(image, name, (left + 6, bottom + 25), font, fontScale, (255, 255, 255), fontThickness)

                    # Save the frame with timestamp in the filename
                    frame_filename = f"matched_frame_{name}_{timestamp}.jpg".replace(":", "-").replace(" ", "_")
                    print(f"Saving image to: {frame_filename}")  # Debugging image saving
                    cv2.imwrite(frame_filename, image)

    # Increment frame count
    frame_count += 1

    # Display the frame
    cv2.imshow("Video", image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
