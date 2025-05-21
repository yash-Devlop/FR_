import time
from fastapi import HTTPException
from db_config import get_db_connection

# In-memory cache dictionary
subscription_status_cache = {}

def fill_company_status():
    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT Companies.Cname, Subscriptions.status
            FROM Companies
            JOIN Subscriptions ON Companies.Cid = Subscriptions.Cid;
        """)
        results = cursor.fetchall()

        if not results:
            return

        for cname, status in results:
            subscription_status_cache[cname] = status

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error filling status: {e}")
    finally:
        conn.close()


def get_status(company_name):

    if company_name in subscription_status_cache:
        return subscription_status_cache[company_name]
    else:
        conn = get_db_connection("super")
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT Subscriptions.Status
                FROM Companies
                JOIN Subscriptions ON Companies.Cid = Subscriptions.Cid
                WHERE Companies.Cname = ?;
            """, (company_name,))
            row = cursor.fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail="Company not found or subscription missing")

            subscription_status_cache[company_name] = row[0]
            return row[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot find company status: {e}")
        finally:
            conn.close()
