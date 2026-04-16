"""Initialize SQLite database for Task 16."""

import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# users table
cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)")
users_data = [
    (1, "Alice", 30, "Berlin"),
    (2, "Bob", 25, "Berlin"),
    (3, "Charlie", 35, "Paris"),
    (4, "Dave", 40, "Berlin"),
    (5, "Eve", 28, "New York"),
]
cursor.executemany("INSERT INTO users VALUES (?,?,?,?)", users_data)

# orders table
cursor.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)")
orders_data = [
    (1, 1, 100.0, "completed"),
    (2, 2, 50.0, "pending"),
    (3, 1, 75.0, "completed"),
    (4, 3, 200.0, "completed"),
    (5, 4, 120.0, "cancelled"),
]
cursor.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders_data)

conn.commit()
conn.close()
