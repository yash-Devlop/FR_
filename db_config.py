import pyodbc

def get_db_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=MOHIT\SQLEXPRESS;"
        "DATABASE=FaceRecognitionDB;"
        "UID=sa;"
        "PWD=Mohit@123;"
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