from fastapi import FastAPI, HTTPException, File, UploadFile
from face_recognition import detect_faces, reg_face
from db_config import get_db_connection
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.post("/register_user")
async def register_user(user_type: str, name: str, val2: str, file: UploadFile = File(...)):
    user_type = user_type
    image_contents = await file.read()
    name = name
    val2 = val2
    
    stored_embeddings, msg = reg_face(image_contents)

    if stored_embeddings is None:
        raise HTTPException(status_code=400, detail=msg)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if user_type == "Employee":
            cursor.execute("INSERT INTO Employees (Name, Department, Image) VALUES (?, ?, ?)", 
                       (name, val2, stored_embeddings))
        elif user_type == "Visitor":
            cursor.execute("INSERT INTO Visitors (Name, Type, Image) VALUES (?, ?, ?)", 
                       (name, val2, stored_embeddings))
        conn.commit()

        return {"message": f"User '{name}' registered successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))



# @app.post("/register")
# async def register_face(name: str, file: UploadFile = File(...)):
#     contents = await file.read()
#     stored_embeddings, msg = reg_face(contents)

#     if stored_embeddings is None:
#         raise HTTPException(status_code=400, detail=msg)

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     try:
#         cursor.execute("INSERT INTO registered_faces (name, embedding) VALUES (?, ?)", 
#                        (name, stored_embeddings))
#         conn.commit()
#     except Exception as e:
#         conn.rollback()
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         conn.close()

#     return {"message": f"Face '{name}' registered successfully"}


@app.post("/match")
async def match_face(file: UploadFile = File(...)):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces(frame)

    if not faces:
        return {"matches": []}

    (x, y, w, h) = faces[0]  # Get face bounding box
    face_crop = frame[y:y+h, x:x+w]  # Crop the detected face

    try:
        embeddings = []
        for _ in range(3):  # Generate 3 embeddings for better matching
            emb = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            emb = np.array(emb, dtype=np.float32)
            embeddings.append(emb)

        current_embedding = np.median(np.array(embeddings), axis=0)
        current_embedding = current_embedding / np.linalg.norm(current_embedding)  # Normalize
        current_embedding = current_embedding.reshape(1, -1)
    except Exception as e:
        return {"matches": ["unknown"]}

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Name, 'Employee' AS UserType, Department AS Type, Image FROM Employees UNION SELECT Name, 'Visitor' AS UserType, Type, Image FROM Visitors")
    registered = cursor.fetchall()
    conn.close()

    best_match = "unknown"
    highest_similarity = -1

    for name, utype, dept, emb_bytes in registered:
        stored_embeddings = np.frombuffer(emb_bytes, dtype=np.float32).reshape(5, -1)

        similarities = cosine_similarity(current_embedding, stored_embeddings)
        max_similarity = np.max(similarities)  

        if max_similarity > highest_similarity:
            highest_similarity = max_similarity
            best_match = {"name": name, "type_or_department": dept, "user_type": utype}

    if highest_similarity < 0.55:  # Threshold to reduce false positives
        return {"matches": ["unknown"]}

    return {"matches": [best_match]}
