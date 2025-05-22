import pyodbc
from fastapi import HTTPException


def get_db_connection(database):
    database = database.lower()
    database = "_".join(database.split(" ")) + "DB"
    server = 'localhost,1433'
    UID = 'sa'
    PWD = 'Mohit@123'

    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={UID};"
        f"PWD={PWD};"
    )

    return pyodbc.connect(connection_string, autocommit=True)


def initialize_db(db_name: str):
    server = 'localhost,1433'
    db_up_name = db_name.lower()
    db_up_name = "_".join(db_up_name.split(" ")) + "DB"

    conn_init = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};",
        autocommit=True
    )

    try:
        cursor_init = conn_init.cursor()
        cursor_init.execute(f"CREATE DATABASE {db_up_name};")
        conn_init.commit()
    except Exception as e:
        raise HTTPException(f"cannot create database: {e}")
    finally: 
        conn_init.close()

    connd = get_db_connection(db_name)
    cursor = connd.cursor()

    try:
        # Employees
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Employees' AND xtype='U')
                CREATE TABLE Employees(
                    EmpId INT PRIMARY KEY IDENTITY (101,1),
                    Name NVARCHAR(50) NOT NULL,
                    Department NVARCHAR(50) NOT NULL,
                    Image VARBINARY(MAX),
                    Status NVARCHAR(50)
                );
        """)

        # EmpAttendance
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='EmpAttendance' AND xtype='U')
                CREATE TABLE EmpAttendance(
                    Id INT PRIMARY KEY IDENTITY(1,1),
                    EmpId INT NOT NULL,
                    EmpName NVARCHAR(50) NOT NULL,
                    EntryTime DATETIME NOT NULL,
                    ExitTime DATETIME
                );
        """)

        # Visitors
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Visitors' AND xtype='U')
                CREATE TABLE Visitors(
                    VisitorId INT PRIMARY KEY IDENTITY (101,1),
                    Name NVARCHAR(50) NOT NULL,
                    Type NVARCHAR(50),
                    Image VARBINARY(MAX),
                    Status NVARCHAR(50)
                );
        """)

        # VisitorEntry
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='VisitorEntry' AND xtype='U')
                CREATE TABLE VisitorEntry(
                    Id INT PRIMARY KEY IDENTITY (1,1),
                    VisitorId INT NOT NULL,
                    VisitorName NVARCHAR(50) NOT NULL,
                    Date DATETIME NOT NULL
                );
        """)

    except Exception as e:
        connd.rollback()
        raise HTTPException(status_code=400, detail=f"Cannot create all tables: {e}")

    finally:
        connd.close()
