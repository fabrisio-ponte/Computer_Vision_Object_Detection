from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

def background_thread():
    while True:
        socketio.emit('message', {'data': 'Hello from server'})
        time.sleep(1)

@app.route('/')
def index():
    return "Hello, Flask-SocketIO is running."

if __name__ == '__main__':
    thread = threading.Thread(target=background_thread)
    thread.daemon = True
    thread.start()
    socketio.run(app, host='0.0.0.0', port=5005)
