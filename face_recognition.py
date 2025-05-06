import cv2
import numpy as np
from deepface import DeepFace
import os


def detect_faces(frame):
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"File not found: {prototxt_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")
    
    print("Loading face detection model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2 - x, y2 - y))
    
    return faces


def reg_face(contents):
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        return None, "Invalid image file"

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces(frame)

    if len(faces) != 1:
        return None, None, "Face not detected or multiple faces detected"

    (x, y, w, h) = faces[0]  # Get face bounding box
    face_crop = frame[y:y+h, x:x+w]  # Crop the detected face

    try:
        embeddings = []
        for _ in range(5):  # Generate 5 embeddings from cropped face
            emb = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            emb = np.array(emb, dtype=np.float32)
            embeddings.append(emb)

        current_embedding = np.median(np.array(embeddings), axis=0)
        current_embedding = current_embedding / np.linalg.norm(current_embedding)  # Normalize
        current_embedding = current_embedding.reshape(1, -1)

        stored_embeddings = np.array(embeddings, dtype=np.float32).tobytes()
        return stored_embeddings, current_embedding,  "Done"
    except Exception as e:
        return None, None, str(e)