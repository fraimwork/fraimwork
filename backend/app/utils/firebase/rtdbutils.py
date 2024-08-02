import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase Admin SDK
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-project.firebaseio.com'
})

# Function to get a reference to the database
def get_database():
    return db.reference()

# Function to read data from the database
def read_data(path):
    ref = get_database().child(path)
    return ref.get()

# Function to write data to the database
def write_data(path, data):
    ref = get_database().child(path)
    ref.set(data)

# Function to update data in the database
def update_data(path, data):
    ref = get_database().child(path)
    ref.update(data)

# Function to delete data from the database
def delete_data(path):
    ref = get_database().child(path)
    ref.delete()