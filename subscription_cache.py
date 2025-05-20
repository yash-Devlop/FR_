import time
from fastapi import HTTPException
from db_config import get_db_connection

# In-memory cache dictionary
subscription_status_cache = {}

def fill_company_status():
    
    conn = get_db_connection("super")
    cursor = conn.cursor()

    cursor.execute("SELECT Cname, Status FROM Companies;")
    C_status = cursor.fetchall()
    print(f"[Cache Refresh] {len(C_status)} companies updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not C_status:
        return

    for i in C_status:
        subscription_status_cache[i[0]] = i[1]


def get_status(company_name):

    if company_name in subscription_status_cache:
        return subscription_status_cache[company_name]
    else:

        conn = get_db_connection("super")
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT Status FROM Companies WHERE Cname = ?;", (company_name,))
            row = cursor.fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail="Company not found")

            subscription_status_cache[company_name] = row[0]
            return row[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot find company status: {e}")
        finally:
            conn.close()
    