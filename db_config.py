import pyodbc
from fastapi import HTTPException
def get_db_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost,1433;"
        "DATABASE=FaceRecognitionDB;"
        "UID=sa;"
        "PWD=Mohit@123;"
        "TrustServerCertificate=yes;"
    )

def initialize_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='registered_faces' AND xtype='U')
        CREATE TABLE registered_faces (
            id INT PRIMARY KEY IDENTITY,
            name NVARCHAR(100),
            embedding VARBINARY(512),  -- 128 * 4 bytes (float32)
            created_at DATETIME DEFAULT GETDATE()
        )
    ''')
    conn.commit()
    conn.close()


def initialize_db_all():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Employees
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'dbo.Employees') AND type = 'U')
            BEGIN
                CREATE TABLE Employees(
                    EmpId INT PRIMARY KEY IDENTITY (1,1),
                    Name NVARCHAR(50) NOT NULL,
                    Department NVARCHAR(50) NOT NULL,
                    Image VARBINARY(MAX),
                    Status NVARCHAR(50)
                );
            END
        """)

        # EmpAttendance
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'dbo.EmpAttendance') AND type = 'U')
            BEGIN
                CREATE TABLE EmpAttendance(
                    Id INT PRIMARY KEY IDENTITY(1,1),
                    EmpId INT NOT NULL,
                    EmpName NVARCHAR(50) NOT NULL,
                    EntryTime DATETIME NOT NULL,
                    ExitTime DATETIME
                );
            END
        """)

        # Visitors
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'dbo.Visitors') AND type = 'U')
            BEGIN
                CREATE TABLE Visitors(
                    VisitorId INT PRIMARY KEY IDENTITY (1,1),
                    Name NVARCHAR(50) NOT NULL,
                    Type NVARCHAR(50),
                    Image VARBINARY(MAX),
                    Status NVARCHAR(50)
                );
            END
        """)

        # VisitorEntry
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'dbo.VisitorEntry') AND type = 'U')
            BEGIN
                CREATE TABLE VisitorEntry(
                    Id INT PRIMARY KEY IDENTITY (1,1),
                    VisitorId INT NOT NULL,
                    VisitorName NVARCHAR(50) NOT NULL,
                    Date DATETIME NOT NULL
                );
            END
        """)
        
        conn.commit()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot create all tables: {e}")
    finally:
        conn.close()


initialize_db_all()