from flask import Flask, render_template
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB URI
db = client['admin']  # Replace with your database name
collection = db['Iris']  # Replace with your collection name

@app.route('/')
def index():
    # Retrieve data from the collection
    data = list(collection.find())  # Convert cursor to list
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)