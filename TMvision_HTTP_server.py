from flask import Flask, request, jsonify, g  # Add 'g' import here
from werkzeug.exceptions import HTTPException
from waitress import serve
import cv2
import numpy as np
import datetime
import socket
import os

app = Flask(__name__)
HOST_NAME = 'TM Vision HTTP Server'
HOST_PORT = 4585

# Utility function to log with timestamp
def log_message(message):
    timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
    print(f'[{timestamp}] {message}')

# Error handler for HTTP exceptions
@app.errorhandler(HTTPException)
def handle_exception(e):
    '''Return HTTP errors.'''
    log_message(str(e))
    return e

# Logging before and after each request
@app.before_request
def before_request():
    # Replacing 'request.remote_port' with a placeholder 'PORT'
    log_message(f'[{request.remote_addr}:PORT] -> {request.method}({request.path}) - Start')
    g.request_start_time = datetime.datetime.utcnow()

@app.after_request
def after_request(response):
    try:
        log_message(f'[{request.remote_addr}:{request.environ.get("REMOTE_PORT")}] -> {request.method}({request.path}) - End')
    except Exception as e:
        log_message(f"Error logging after request: {str(e)}")
    return response


# Routes
@app.route('/', methods=['GET'])
def default():
    return jsonify({"result": "api", "message": "running"})

@app.route('/api/<string:m_method>', methods=['GET'])
def get_method(m_method):
    if m_method == 'status':
        return jsonify({"result": "status", "message": "I'm ok"})
    else:
        return jsonify({"result": "fail", "message": "wrong request"})

@app.route('/api/<string:m_method>', methods=['POST'])
def post_method(m_method):
    model_id = request.args.get('model_id')

    if not model_id:
        log_message('model_id is not set')
        return jsonify({"message": "fail", "result": "model_id required"})
    else:
        log_message('Model_ID : '+model_id)

    # Dummy processing, replace with real image processing using CV2
    img = cv2.imdecode(np.frombuffer(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    Folder_Name = "Output"
    try:
        # Placeholder logic, replace with actual model inference
        if m_method == 'CLS':
            Folder_Name = Folder_Name+"/CLS"
            if not os.path.exists(Folder_Name):
                os.makedirs(Folder_Name)
                print(f"Folder '{Folder_Name}' created.")
            else:
                print(f"Folder '{Folder_Name}' already exists.")
            cv2.imwrite("Output/CLS/output_image.png", img)
            return jsonify({
                "message": "success",
                "result": "NG",
                "score": 0.987
            })
        elif m_method == 'DET':
            Folder_Name = Folder_Name+"/DET"
            if not os.path.exists(Folder_Name):
                os.makedirs(Folder_Name)
                print(f"Folder '{Folder_Name}' created.")
            else:
                print(f"Folder '{Folder_Name}' already exists.")
            cv2.imwrite("Output/DET/output_image.png", img)
            return jsonify({
                "message": "success",
                "annotations": [
                    {"box_cx": 150, "box_cy": 150, "box_w": 100, "box_h": 100, "label": "apple", "score": 0.964, "rotation": -45},
                    {"box_cx": 550, "box_cy": 550, "box_w": 100, "box_h": 100, "label": "car", "score": 1.000, "rotation": 0},
                    {"box_cx": 350, "box_cy": 350, "box_w": 150, "box_h": 150, "label": "mobilephone", "score": 0.886, "rotation": 135}
                ]
            })
        else:
            return jsonify({"message": "no method"})
    except Exception as e:
        log_message(f"Error processing request: {str(e)}")
        return jsonify({"message": "Error processing request", "error": str(e)})

# Entry point
if __name__ == '__main__':
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except Exception as e:
        log_message(str(e))
        host_ip = "127.0.0.1"
    log_message(f'serving on http://{host_ip}:{HOST_PORT}')
    serve(app, host=host_ip, port=HOST_PORT, ident=HOST_NAME)

