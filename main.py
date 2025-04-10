from fastapi import FastAPI, HTTPException, File, UploadFile
from face_recognition import detect_faces, reg_face
from db_config import get_db_connection
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import base64


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/register_user")
async def register_user(user_type: str, name: str, val2: str, file: UploadFile = File(...)):
    user_type = user_type
    image_contents = await file.read()
    name = name
    val2 = val2

    print("User Type", user_type)
    print("Name", name)
    print("Value 2", val2)
    print("Image", type(image_contents))
    
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
        return {"matches": [{"name":"unknown"}]}

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
        return {"matches": [{"name":"unknown"}]}

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT EmpId, Name, 'Employee' AS UserType, Department AS Type, Image, Status FROM Employees UNION SELECT VisitorId, Name, 'Visitor' AS UserType, Type, Image, Status FROM Visitors")
    registered = cursor.fetchall()
    conn.close()

    best_match = "unknown"
    highest_similarity = -1

    for mid, name, utype, dept, emb_bytes, status in registered:
        stored_embeddings = np.frombuffer(emb_bytes, dtype=np.float32).reshape(5, -1)

        similarities = cosine_similarity(current_embedding, stored_embeddings)
        max_similarity = np.max(similarities)  

        if max_similarity > highest_similarity:
            highest_similarity = max_similarity
            best_match = {"id": mid, "name": name, "type_or_department": dept, "user_type": utype, "status":status}
    if not best_match:
        return {"matches": [{"name":"unknown"}]}
    if highest_similarity < 0.55:  # Threshold to reduce false positives
        return {"matches": [{"name":"unknown"}]}


    if not best_match["status"]=="Blocked":
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            if best_match["user_type"] == "Employee":
                cursor.execute("INSERT INTO EmpAttendance (EmpId, EmpName) VALUES (?, ?)", 
                            (best_match["id"], best_match["name"]))
            elif best_match["user_type"] == "Visitor":
                cursor.execute("INSERT INTO VisitorEntry (VisitorId, VisitorName) VALUES (?, ?)", 
                            (best_match["id"], best_match["name"]))
            conn.commit()
        except Exception as e:
            conn.rollback() 
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            conn.close()

    return {"matches": [best_match]}


@app.post("/get_all_employees")
async def get_all_employees():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT EmpId, Name, Department, Status, Image FROM Employees")
    employees = cursor.fetchall()
    conn.close()

    return {
        "employees": [
            {
                "id": emp[0],
                "name": emp[1],
                "department": emp[2],
                "status": emp[3],
                "image": base64.b64encode(emp[4]).decode('utf-8') if emp[4] else None
            }
            for emp in employees
        ]
    }

@app.post("/get_all_visitors")
async def get_all_visitors():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT VisitorId, Name, Type, Status, Image FROM Visitors")
    visitors = cursor.fetchall()
    conn.close()

    return {
        "visitors": [
            {
                "id": visitor[0],
                "name": visitor[1], 
                "type": visitor[2], 
                "status": visitor[3],
                "image": base64.b64encode(visitor[4]).decode('utf-8') if visitor[4] else None
            }
            for visitor in visitors
        ]
    }

@app.post("/update_employee_status")
async def update_employee_status(emp_id: int, name: str, department: str, status: str):
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE Employees SET Name = ?, Department = ?, Status = ? WHERE EmpId = ?",
            (name, department, status, emp_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Employee not found")
        conn.commit()
        return {"message": f"Employee with ID {emp_id} updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.post("/update_visitor_status")
async def update_visitor_status(visitor_id: int, name: str, visitor_type: str, status: str):
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE Visitors SET Name = ?, Type = ?, Status = ? WHERE VisitorId = ?",
            (name, visitor_type, status, visitor_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Visitor not found")
        conn.commit()
        return {"message": f"Visitor with ID {visitor_id} updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/delete_employee")
async def delete_employee(emp_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM Employees WHERE EmpId = ?", (emp_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Employee not found")
        conn.commit()
        return {"message": f"Employee with ID {emp_id} deleted successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.post("/delete_visitor")
async def delete_visitor(visitor_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM Visitors WHERE VisitorId = ?", (visitor_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Visitor not found")
        conn.commit()
        return {"message": f"Visitor with ID {visitor_id} deleted successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()