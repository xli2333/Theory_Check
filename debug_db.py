import sqlite3
import os

def inspect_db(path):
    print(f'--- Inspecting {path} ---')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        print(f'Tables: {tables}')
        for t in tables:
            t_name = t[0]
            c.execute(f"PRAGMA table_info({t_name})")
            cols = c.fetchall()
            print(f'Table {t_name} columns: {[col[1] for col in cols]}')
            
            if t_name in ['theories', 'knowledge_points']:
                 try:
                     c.execute(f"SELECT DISTINCT category FROM {t_name}")
                     cats = c.fetchall()
                     print(f'Categories in {t_name}: {cats}')
                 except:
                     print(f"Could not select categories from {t_name}")

        conn.close()
    except Exception as e:
        print(f'Error: {e}')

inspect_db('knowledge_base.db')
inspect_db('knowledge_base_v2.db')
