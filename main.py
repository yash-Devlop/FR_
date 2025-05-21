import base64
import bcrypt
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from face_recognition import detect_faces, reg_face
from db_config import get_db_connection, initialize_db
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from typing import Any, Optional, get_type_hints
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
from scheduler import subscription_reminder_task
from subscription_cache import fill_company_status, get_status
from apscheduler.schedulers.background import BackgroundScheduler

class Company_register(BaseModel):
    Username: str
    Cname: str
    Cpass: str
    Caddress: str
    Cphone: str
    state: str
    city: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    fill_company_status()
    scheduler.add_job(subscription_reminder_task, 'cron', hour=23, minute=55)
    scheduler.add_job(fill_company_status, 'interval', minutes=30)
    scheduler.start()

    yield

    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/super/register_company")
async def register_company(form: Company_register):
    for field_name, field_type in get_type_hints(Company_register).items():
        if field_type == str:
            value = getattr(form, field_name)
            if not value.strip():
                return {"status": "bad", "matches": f"invalid-{field_name}"}
            
    if not form.Cphone.startswith("+91"):
        return {"status": "bad", "matches": "invalid-Cphone-prefix"}

    if form.Cname.lower() == 'super':
        return {"status": "bad", "matches": "cannot-create-company-with-this-name"}

    time = datetime.now(ZoneInfo("Asia/Kolkata"))

    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:

        company_id = str(uuid.uuid4()) 
        hashed_password = bcrypt.hashpw(form.Cpass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        initialize_db(form.Cname)
        cursor.execute("""
            INSERT INTO Companies (Cid, Username, Cname, Cpass, Caddress, Cphone, City, State, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (company_id, form.Username, form.Cname, hashed_password, form.Caddress, form.Cphone, form.city, form.state, time))

        conn.commit()
        return {"status": "ok", "detail": "company-registered"}
        
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=400, detail=f"cannot add company details: {e}")
    
    finally:
        if conn:
            conn.close()


@app.post("/super/get_all_companies")
async def get_all_companies():

    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM Companies;")
        companies = cursor.fetchall()

        if not companies:
            return {"status": "ok", "matches": "no-company-found"}
        
        return {
            "status": "ok",
            "matches": [
                {
                    "Cid": company[0],
                    "Username": company[1],
                    "Cname": company[2],
                    "Caddress": company[4],
                    "Cphone": company[5],
                    "city": company[6],
                    "state": company[7],
                    "created_at": company[8]
                }
                for company in companies
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot get all companies: {e}")
    finally:
        conn.close()


@app.post("/super/change_company_password")
async def change_company_password(Cid: str, new_pass: str):

    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        
        hashed_password = bcrypt.hashpw(new_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        cursor.execute("""
            UPDATE Companies
            SET Cpass = ?
            WHERE Cid = ?
        """, (hashed_password, Cid))

        return {"status": "ok", "matches": "password-changed"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"error updating password: {e}")
    finally: 
        conn.close()

@app.post("/super/create_subscription")
async def handle_subscription(Cid: str, duration: int):

    if not 1 <= duration <= 60:
        return {"status": "bad", "matches": "duration-invalid"}

    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        start_dt = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        original_end_dt = start_dt + relativedelta(months=duration) - timedelta(days=1)
        final_end_dt = original_end_dt

        cursor.execute("SELECT * FROM Companies WHERE Cid = ?", (Cid,))
        company = cursor.fetchone()
        if not company:
            return {"status": "bad", "matches": "companyid-not-found"}

        cursor.execute("""
            SELECT Subid, Enddate FROM Subscriptions
            WHERE Cid = ?
        """, (Cid,))
        existing = cursor.fetchone()

        if existing:
            existing_subid = existing[0]
            existing_end = existing[1].date()
            today = start_dt

            if existing_end >= today:
                remaining_days = (existing_end - today).days + 1
                final_end_dt += timedelta(days=remaining_days)

            cursor.execute("DELETE FROM Subscriptions WHERE Subid = ?", (existing_subid,))

        cursor.execute("SELECT MAX(Subid) FROM Subscriptions")
        max_id = cursor.fetchone()[0] or 0
        new_id = max_id + 1

        cursor.execute("""
            INSERT INTO Subscriptions (Subid, Cid, Cname, Startdate, Duration, Enddate, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (new_id, Cid, company[2], start_dt, duration, final_end_dt, 'active'))

        cursor.execute("""
            INSERT INTO SubscriptionHistory (Subid, Cid, Cname, Startdate, Duration, Enddate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (new_id, Cid, company[2], start_dt, duration, original_end_dt))

        conn.commit()

        sub_detail = {
            "subid": new_id,
            "Cname": company[2],
            "start_date": start_dt,
            "original_end_date": original_end_dt,
            "final_end_date": final_end_dt,
            "duration": duration
        }

        return {"status": "ok", "details": sub_detail}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"Subscription error: {e}")
    finally:
        conn.close()


@app.post("/super/get_all_subscriptions")
async def get_all_subscriptions():

    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM Subscriptions")
        subscriptions = cursor.fetchall()

        if not subscriptions:
            {"status": "ok", "matches": "no-subscription-found"}

        return [
            {
                "Subid": subscription[0],
                "Cid": subscription[1],
                "Cname": subscription[2],
                "strt_date": subscription[3],
                "duration": subscription[4],
                "end_date": subscription[5],
                "status": subscription[6]
            }

            for subscription in subscriptions
        ]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot fetch data: {e}")
    finally:
        conn.close()

@app.post("/super/get_all_subscriptions_by_id")
async def get_all_subscriptions_by_id(Cid: str):

    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM SubscriptionHistory WHERE Cid = ?", (Cid,))
        subscriptions = cursor.fetchall()

        if not subscriptions:
            return{"status": "bad", "matches": "cannot-find-company-details"}

        return {
            "status": "ok",
            "data": {
                "Cname": subscriptions[0][3],
                "details": [
                    {
                        "Cid": subscription[2],
                        "startdate": subscription[4],
                        "duration": subscription[5],
                        "enddate": subscription[6]
                    }
                    for subscription in subscriptions
                ]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {e}")
    finally:
        conn.close()

@app.post("/super/update_subscriptions")
async def update_subscriptions(Subid: int, status: str):
    if status not in ['Active', 'Blocked']:
        return {"status": "bad", "matches": "Active-and-Blocked-only-allowed"}

    conn = get_db_connection("super")
    cursor = conn.cursor()

    try: 
        cursor.execute("""
            UPDATE Subscriptions
            SET Status = ?
            WHERE Subid = ?
        """, (status, Subid))

        if cursor.rowcount == 0:
            return {"status": "bad", "matches": "Subscription ID not found"}

        conn.commit()
        return {"status": "ok", "matches": f"Subscription {Subid} updated to {status}"}        

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"Cannot update status: {e}")
    finally:
        conn.close()


@app.post("/check_subscription")
async def check_expiring_subscriptions(Cid: str):
    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT Status FROM Subscriptions WHERE Cid = ?", (Cid,))
        sub_status = cursor.fetchone()

        if not sub_status:
            return {"status": "bad", "messaage": "cannot-find-subscription"}
        
        if sub_status[0] == 'active':
            return {"status": "ok", "matches": "active"}
        else:
            return {"status": "bad", "matches": "not-active"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking expirations: {e}")
    finally:
        conn.close()


@app.post("/register_user")
async def register_user(Cname: str, user_type: str, name: str, val2: str, file: UploadFile = File(...)):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}
    
    data = {
        "Cname": Cname,
        "user_type": user_type,
        "name": name,
        "val2": val2
    }

    for key, value in data.items():
        if not value.strip():
            return {"status": "bad", "matches": f"invalid-{key}"}

    image_contents = await file.read()
    stored_embeddings, current_embedding, msg = reg_face(image_contents)

    if stored_embeddings is None:
        raise HTTPException(status_code=400, detail=msg)

    conn = get_db_connection(Cname)
    cursor = conn.cursor()

    try:
        def check_table(table_name, embedding_col, threshold=0.55):
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            highest_similarity = -1

            for row in rows:
                emb_bytes = row[3]
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
            
            return highest_similarity

        emp_similarity = check_table("Employees", "Image")
        if emp_similarity >= 0.55:
            return {"status": "registered", "matches": "already-employee"}

        visitor_similarity = check_table("Visitors", "Image")
        if visitor_similarity >= 0.55:
            return {"status": "registered", "matches": "already-visitor"}

        if user_type == "Employee":
            cursor.execute(
                "INSERT INTO Employees (Name, Department, Image, Status) VALUES (?, ?, ?, ?)", 
                (name, val2, stored_embeddings, 'Allowed')
            )
            conn.commit()
            return {"status": "ok", "matches": f"Employee {name} successfully registered"}

        elif user_type == "Visitor":
            cursor.execute(
                "INSERT INTO Visitors (Name, Type, Image) VALUES (?, ?, ?)",
                (name, val2, stored_embeddings)
            )
            conn.commit()
            return {"status": "ok", "matches": f"Visitor {name} successfully registered"}

        else:
            return {"status": "bad", "matches": "invalid user_type"}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()



@app.post("/entry_match")
async def match_face_entry(Cname: str, file: UploadFile = File(...)):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces(frame)

    if not faces:
        return {"status": "unknown", "matches": {"name":"unknown"}}

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
        return {"status": "unknown", "matches": {"name":"unknown"}}

    conn = get_db_connection(Cname)
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
        return {"status": "unknown", "matches": {"name":"unknown"}}
    if highest_similarity < 0.55:  # Threshold to reduce false positives
        return {"status": "unknown", "matches": {"name":"unknown"}}

    time = datetime.now(ZoneInfo("Asia/Kolkata"))

    if best_match["status"]!="Blocked":
        conn = get_db_connection(Cname)
        cursor = conn.cursor()
        try:
            if best_match["user_type"] == "Employee":

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
                    return {"status": "entry_exists", "matches": "Open Entry already exists"}

            elif best_match["user_type"] == "Visitor":
                two_hours_ago = time - timedelta(hours=2)

                cursor.execute("""
                    SELECT TOP 1 Date FROM VisitorEntry
                    WHERE VisitorId = ?
                    ORDER BY Date DESC
                """, (best_match["id"],))
                last_entry = cursor.fetchone()

                if last_entry is not None:
                    last_entry_time = last_entry[0]
                    if isinstance(last_entry_time, str):
                        last_entry_time = datetime.fromisoformat(last_entry_time)
                    if last_entry_time > two_hours_ago:
                        return {"status": "already-entered", "matches": f"Visitor {best_match['name']} entered within last 2 hours"}

                cursor.execute("""
                    INSERT INTO VisitorEntry (VisitorId, VisitorName, Date) VALUES (?, ?, ?)
                """, (best_match["id"], best_match["name"], time))

            conn.commit()
            return {"status": "ok", "matches": [best_match]}
        except Exception as e:
            conn.rollback() 
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            conn.close()


@app.post("/exit_match") #Type = Exit
async def match_face_exit(Cname: str, file: UploadFile = File(...)):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces(frame)

    if not faces:
        return {"status": "unknown", "matches": {"name":"unknown"}}

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
        return {"status": "unknown", "matches": {"name":"unknown"}}

    conn = get_db_connection(Cname)
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
        return {"status": "unknown", "matches": {"name":"unknown"}}
    if highest_similarity < 0.55:  # Threshold to reduce false positives
        return {"status": "unknown", "matches": {"name":"unknown"}}

    time = datetime.now(ZoneInfo("Asia/Kolkata"))

    if best_match["status"]!="Blocked":
        conn = get_db_connection(Cname)
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

                if entered_employee:
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
                    return {"status": "entry_not_found", "matches": "No latest Entry Found"} #If no entry found
            
            elif best_match["user_type"] == "Visitor":
                return {"status": "ok", "matches": f"visitor {best_match['name']} exited"}
            conn.commit()
        except Exception as e:
            conn.rollback() 
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            conn.close()

    return {"status": "ok", "matches": best_match}


@app.post("/get_all_employees")
async def get_all_employees(Cname: str):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    conn = get_db_connection(Cname)
    cursor = conn.cursor()
    cursor.execute("SELECT EmpId, Name, Department, Status, Image FROM Employees")
    employees = cursor.fetchall()
    conn.close()

    return {
        "status": "ok", 
        "employees": [
            {
                "id": emp[0],
                "name": emp[1],
                "department": emp[2],
                "status": emp[3],
                "image": base64.b64encode(emp[4]).decode('utf-8') if emp[4] else None  #<img src={`data:image/jpeg;base64,${image}`} alt="Employee" />
            }
            for emp in employees
        ]
    }

@app.post("/get_all_visitors")
async def get_all_visitors(Cname: str):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    conn = get_db_connection(Cname)
    cursor = conn.cursor()
    cursor.execute("SELECT VisitorId, Name, Type, Status, Image FROM Visitors")
    visitors = cursor.fetchall()
    conn.close()

    return {
        "status": "ok",
        "visitors": [
            {
                "Vid": visitor[0],
                "Vname": visitor[1], 
                "Vtype": visitor[2], 
                "Vstatus": visitor[3],
                "Vimage": base64.b64encode(visitor[4]).decode('utf-8') if visitor[4] else None
            }
            for visitor in visitors
        ]
    }

@app.post("/update_employee_status")
async def update_employee_status(Cname: str, emp_id: int, name: str, department: str, status: str):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}
    
    if status not in ['Allowed', 'Blocked']:
        return {"status": "bad", "matches": "Allowed-and-Blocked-only"}

    conn = get_db_connection(Cname)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE Employees SET Name = ?, Department = ?, Status = ? WHERE EmpId = ?",
            (name, department, status, emp_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Employee not found")
        conn.commit()
        return {"status": "ok", "matches": f"Employee with ID {emp_id} updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.post("/update_visitor_status")
async def update_visitor_status(Cname: str, visitor_id: int, name: str, visitor_type: str, status: str):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"} 
    
    conn = get_db_connection(Cname)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE Visitors SET Name = ?, Type = ?, Status = ? WHERE VisitorId = ?",
            (name, visitor_type, status, visitor_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Visitor not found")
        conn.commit()
        return {"status": "ok", "matches": f"Visitor with ID {visitor_id} updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/delete_employee")
async def delete_employee(Cname: str, emp_id: int):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    conn = get_db_connection(Cname)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM Employees WHERE EmpId = ?", (emp_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Employee not found")
        conn.commit()
        return {"status": "ok", "matches": f"Employee with ID {emp_id} deleted successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.post("/delete_visitor")
async def delete_visitor(Cname:str, visitor_id: int):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    conn = get_db_connection(Cname)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM Visitors WHERE VisitorId = ?", (visitor_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Visitor not found")
        conn.commit()
        return {"status": "ok", "matches": f"Visitor with ID {visitor_id} deleted successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.post("/super_login")
async def login(user_id: str = Form(...), password: str = Form(...)):
    try:
        with open("/home/ubuntu/FR_/credentials.txt", "r") as file:
            credentials = file.readlines()

        for line in credentials:
            stored_id, stored_password = line.strip().split(":")

        if user_id == stored_id and password == stored_password:
            return {"status": "ok", "matches": "Login successful", "role": "super-admin"}

        return{"status": "bad", "matches": "invalid-credentials"}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Credentials file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/company_login")
async def company_login(Username: str, password: str):
    conn = get_db_connection("super")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * from companies WHERE Userid = ?", (Username,))
        User = cursor.fetchone()

        if not User:
            return {"status": "bad", "matches": "company-not-found"}
        
        hashed_password = User[3]

        if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            return {"staus": "bad", "matches": "invalid-credentials"} 
        
        details = {
            "Cid": User[0],
            "Userid": User[1],
            "Cname": User[2],
        }

        return {"status": "ok", "matches": details, "role": "company-admin"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot login: {e}")
    finally:
        conn.close()

@app.post("/get_all_attendance")
async def get_all_attendance(Cname: str):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    conn = get_db_connection(Cname)
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
async def emp_hours_worked(Cname: str, EmpId: int, start_date: Optional[str] = Query(None), end_date: Optional[str] = Query(None)) -> dict[str, Any]:

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    conn = get_db_connection(Cname)
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
            return {"status": "unknown", "name": "Unknown", "details": []}

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

        return {"status": "ok", "name": emp_name, "details": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/get_visitor_data")
async def get_visitor_data(Cname: str, VisitorId: int):

    if get_status(Cname) == "blocked":
        return {"status": "blocked", "matches": "company-blocked"}

    conn = get_db_connection(Cname)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM VisitorEntry WHERE VisitorId = ?;", (VisitorId,))
        visitor = cursor.fetchone()

        if visitor:
            return {
                "status": "ok",
                "matches": {
                        "id": visitor[0],
                        "Vid": visitor[1],
                        "Vname": visitor[2],
                        "Entry_time": visitor[3]
                }
            }
        else:
            return({"status": "bad","matches": "data not found"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching visitor data: {e}")
    finally:
        conn.close()
