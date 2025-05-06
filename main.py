from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from face_recognition import detect_faces, reg_face
from db_config import get_db_connection
import cv2 
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import base64
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any, Optional
from zoneinfo import ZoneInfo

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
  
    stored_embeddings, current_embedding, msg = reg_face(image_contents)

    if stored_embeddings is None:
        raise HTTPException(status_code=400, detail=msg)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:

        if user_type == "Employee":

            cursor.execute("SELECT * FROM Employees;")
            registered = cursor.fetchall()

            highest_similarity = -1
            best_match = None
            if registered:
                for mid, Ename, dept, emb_bytes, status in registered:
                    if emb_bytes is None:
                        continue

                    try:
                        registered_embeddings = np.frombuffer(emb_bytes, dtype=np.float32).reshape(5, -1)
                    except ValueError:
                        continue

                    similarities = cosine_similarity(current_embedding, registered_embeddings)
                    max_similarity = np.max(similarities)

                    if max_similarity > highest_similarity:
                        highest_similarity = max_similarity
                        best_match = {"id": mid, "name": Ename, "type_or_department": dept, "status":status}

            if not best_match or highest_similarity < 0.55:

                cursor.execute("INSERT INTO Employees (Name, Department, Image) VALUES (?, ?, ?)", 
                        (name, val2, stored_embeddings))
            
                conn.commit()                
                return {"matches": f"Employee {name} succesfully registered"}
            else:
                return {"matches": "already registered"}
        elif user_type == "Visitor":
            cursor.execute("INSERT INTO Visitors (Name, Type, Image) VALUES (?, ?, ?)", 
                       (name, val2, stored_embeddings))
            conn.commit()
            return {"macthes": f"Visitor {name} registered succesfully"}
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


@app.post("/entry_match") #Type = Entry
async def match_face_entry(file: UploadFile = File(...)):
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

    best_match = "unknown"
    highest_similarity = -1

    for mid, name, utype, dept, emb_bytes, status in registered:
        if emb_bytes is None:
            continue

        try:
            stored_embeddings = np.frombuffer(emb_bytes, dtype=np.float32).reshape(5, -1)
        except ValueError:
            continue

        similarities = cosine_similarity(current_embedding, stored_embeddings)
        max_similarity = np.max(similarities)  

        if max_similarity > highest_similarity:
            highest_similarity = max_similarity
            best_match = {"id": mid, "name": name, "type_or_department": dept, "user_type": utype, "status":status}
    if not best_match:
        return {"matches": [{"name":"unknown"}]}
    if highest_similarity < 0.55:  # Threshold to reduce false positives
        return {"matches": [{"name":"unknown"}]}

    time = datetime.now(ZoneInfo("Asia/Kolkata"))

    if best_match["status"]!="Blocked":
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            if best_match["user_type"] == "Employee":
                
                # Check if there is already an open attendance record
                cursor.execute("""
                    SELECT TOP 1 1 FROM EmpAttendance 
                    WHERE EmpId = ? AND ExitTime IS NULL 
                    ORDER BY EntryTime DESC
                """, (best_match["id"],))

                if cursor.fetchone() is None:

                    cursor.execute("""
                        INSERT INTO EmpAttendance (EmpId, EmpName, EntryTime) 
                        VALUES (?, ?, ?)
                    """, (best_match["id"], best_match["name"], time))
                else:
                    return {"matches": "Open Entry already exists"}

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


@app.post("/exit_match") #Type = Exit
async def match_face_exit(file: UploadFile = File(...)):
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
        if emb_bytes is None:
            continue

        try:
            stored_embeddings = np.frombuffer(emb_bytes, dtype=np.float32).reshape(5, -1)
        except ValueError:
            continue

        similarities = cosine_similarity(current_embedding, stored_embeddings)
        max_similarity = np.max(similarities)  

        if max_similarity > highest_similarity:
            highest_similarity = max_similarity
            best_match = {"id": mid, "name": name, "type_or_department": dept, "user_type": utype, "status":status}
    if not best_match:
        return {"matches": [{"name":"unknown"}]}
    if highest_similarity < 0.55:  # Threshold to reduce false positives
        return {"matches": [{"name":"unknown"}]}

    time = datetime.now(ZoneInfo("Asia/Kolkata"))

    if not best_match["status"]=="Blocked":
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            if best_match["user_type"] == "Employee":
                cursor.execute("""
                    SELECT Id 
                    FROM EmpAttendance 
                    WHERE EmpId = ? 
                    AND CONVERT(date, EntryTime) = CONVERT(date, GETDATE()) 
                    AND ExitTime IS NULL;
                """, (best_match["id"],))

                entered_employee = cursor.fetchone()

                if entered_employee is not None:
                    cursor.execute("""
                        UPDATE EmpAttendance
                        SET ExitTime = ?
                        WHERE Id = (
                            SELECT TOP 1 Id
                            FROM EmpAttendance
                            WHERE EmpId = ? AND ExitTime IS NULL
                            ORDER BY EntryTime DESC);
                        """, (time, best_match["id"]))
                else:
                    return {"matches": "No latest Entry Found"} #If no entry found
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


@app.post("/login")
async def login(user_id: str = Form(...), password: str = Form(...)):
    try:
        with open("/home/ubuntu/FR_/credentials.txt", "r") as file:
            credentials = file.readlines()

        for line in credentials:
            stored_id, stored_password = line.strip().split(":")

        if user_id == stored_id and password == stored_password:
            return {"message": "Login successful", "role": "Admin"}

        raise HTTPException(status_code=401, detail="Invalid credentials")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Credentials file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_all_attendance")
async def get_all_attendance():
    
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM EmpAttendance;")

        all_attendance = cursor.fetchall()

        return [
            {
                "Id": attendance[0],
                "EmpId": attendance[1],
                "EmpName": attendance[2],
                "EntryTime": attendance[3],
                "ExitTime": attendance[4]
            }
            for attendance in all_attendance
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching EmpAttendance: {e}")
    finally:
        conn.close()


@app.post("/get_employee_hours_worked_by_date") #{"name": name, "details": [{'date': '2025-04-22', 'entry': ['09:00', '13:30'], 'exit': ['12:00', '18:00'], 'hours': hh:mm}, ...]}
async def emp_hours_worked(EmpId: int, start_date: Optional[str] = Query(None), end_date: Optional[str] = Query(None)) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        today = datetime.now(ZoneInfo("Asia/Kolkata"))

        if start_date and end_date:
            start_dt = datetime.strptime(start_date, "%d-%m-%Y")
            end_dt = datetime.strptime(end_date, "%d-%m-%Y")

            if end_dt > today:
                end_dt = today

        else:
            end_dt = today
            start_dt = today - timedelta(days=30)


        cursor.execute("""
            SELECT 
                e.Name AS EmpName,
                CAST(a.EntryTime AS DATE) AS AttendanceDate,
                a.EntryTime,
                a.ExitTime
            FROM 
                EmpAttendance a
            JOIN 
                Employees e ON a.EmpId = e.EmpId
            WHERE 
                a.EmpId = ? AND 
                a.EntryTime BETWEEN ? AND ?
            ORDER BY 
                AttendanceDate, a.EntryTime
        """, (EmpId, start_dt, end_dt + timedelta(days=1)))

        rows = cursor.fetchall()

        if not rows:
            return {"name": "Unknown", "details": []}

        attendance = defaultdict(lambda: {'entry': [], 'exit': [], 'hours': timedelta()})
        emp_name = rows[0].EmpName

        for row in rows:
            date_str = row.AttendanceDate.strftime('%Y-%m-%d')
            entry_time = row.EntryTime.strftime('%H:%M')
            if row.ExitTime:
                exit_time = row.ExitTime.strftime('%H:%M')
                attendance[date_str]['entry'].append(entry_time)
                attendance[date_str]['exit'].append(exit_time)
                attendance[date_str]['hours'] += row.ExitTime - row.EntryTime

        result = []
        for date, info in attendance.items():
            total_seconds = info['hours'].total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)

            result.append({
                'date': date,
                'entry': info['entry'],
                'exit': info['exit'],
                'hours': f"{hours:02d}:{minutes:02d}"
            })

        return {"name": emp_name, "details": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
