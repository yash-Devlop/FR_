from datetime import datetime
from zoneinfo import ZoneInfo
from db_config import get_db_connection


def subscription_reminder_task():
    conn = get_db_connection("super")
    cursor = conn.cursor()

    try:
        now = datetime.now(ZoneInfo("Asia/Kolkata")).date()

        print(f"\n[{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}] Subscription Reminder Check")

        cursor.execute("SELECT Subid, Enddate FROM Subscriptions")
        ed_date = cursor.fetchall()

        all_dates = [{"subid": row[0], "date": row[1].date()} for row in ed_date]

        for date in all_dates:
            if now == date['date']:
                cursor.execute("""
                    UPDATE Subscriptions
                    SET Status = 'blocked'
                    WHERE Subid = ?            
                """, (date['subid'],))

        conn.commit()
        

        cursor.execute("""
            SELECT s.Subid, s.Cid, c.Cname, s.Enddate
            FROM Subscriptions s
            JOIN Companies c ON s.Cid = c.Cid
            WHERE s.Status = 'active'
        """)
        results = cursor.fetchall()

        for subid, cid, cname, end_date in results:
            if end_date is None:
                continue

            days_left = (end_date.date() - now).days

            if 0 <= days_left <= 30 and days_left % 5 == 0:
                if days_left == 0:
                    print(f"'{cname}' subscription has expired today ({end_date.strftime('%Y-%m-%d')})")
                else:
                    print(f"'{cname}' subscription expires in {days_left} days (on {end_date.strftime('%Y-%m-%d')})")

    except Exception as e:
        print(f"Reminder check error: {e}")
    finally:
        conn.close()


def check_status(Cname: str, status: str, status_list = {}):

    status_list[Cname] = status
    return status_list
