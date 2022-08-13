from flask import Flask, render_template, url_for, Response, request

import cv2
import threading
import numpy as np
import webbrowser
import datetime, time
import os, sys
from threading import Thread
import tensorflow as tf
from keras.preprocessing import image


app = Flask(__name__, template_folder='./templates')

# url to use phone camera
#url = "http://192.168.5.107:8080/video"
lock = threading.Lock()
camera = cv2.VideoCapture(0)         # Replace url = 0 if not using mobile camera

# Load trained model
model = tf.keras.models.load_model('../Model/model_csv.h5')
label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
face_haar_cascade = cv2.CascadeClassifier('../Model/haarcascade_frontalface_default.xml')
final_pred = []

global capture
capture=0

# Code related to play song in chrome
url = 'https://www.youtube.com/results?search_query='
chrome_path = '/usr/bin/google-chrome %s'
playlist = ''

def gen_frames():  # generate frame by frame from camera
    global capture
    with lock:
        while True:
            success, frame = camera.read()
            if success:
                if(capture):
                    capture=0
                    cam_img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_haar_cascade.detectMultiScale(cam_img_gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        roi_gray = cam_img_gray[y:y+h, x:x+w]
                        roi_gray = cv2.resize(roi_gray, (48,48))
                        img_pixels = image.img_to_array(roi_gray)
                        img_pixels = np.expand_dims(img_pixels, axis=0)

                        predictions = model.predict(img_pixels)
                        emotion_label = np.argmax(predictions)
                        emotion_prediction = label_dict[emotion_label]
                        emotion = emotion_prediction
                        print('---------' + emotion_prediction + '----------')

                        if emotion_prediction == 'Neutral':
                            playlist = 'new+songs'
                            webbrowser.get(chrome_path).open(url+playlist)
                        if emotion_prediction == 'Happiness':
                            playlist = 'happy+songs'
                            webbrowser.get(chrome_path).open(url+playlist)
                        if emotion_prediction == 'Angry':
                            playlist = 'angry+songs'
                            webbrowser.get(chrome_path).open(url+playlist)
                try:                    
                    scale_percent = 40.0       # Percent by which the image is resized
                    width = int(frame.shape[1] * scale_percent / 100)
                    height = int(frame.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)        # resize image

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except:
                    return          
            else:
                pass
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
            
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

cv2.destroyAllWindows()



#-------------- Code of integrating OpenCv and Flask --------------

# from flask import Flask, render_template, url_for, Response, request

# # Libraries for using phone camera
# import requests
# import cv2
# import numpy as np
# import imutils

# #import cv2
# #import numpy as np
# import datetime, time
# import os, sys
# from threading import Thread


# app = Flask(__name__, template_folder='./templates')

# url = "http://192.168.5.105:8080/video"         # url to use phone camera
# camera = cv2.VideoCapture(url)

# global capture
# capture=0

# #make shots directory to save pics
# try:
#     os.mkdir('./shots')
# except OSError as error:
#     pass
    
# def gen_frames():  # generate frame by frame from camera
#     global capture
#     while True:
#         success, frame = camera.read()
#         if success: 
#             if(capture):
#                 capture=0
#                 now = datetime.datetime.now()
#                 p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
#                 cv2.imwrite(p, frame)
#             try:
                
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             except Exception as e:
#                 pass
                
#         else:
#             pass

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/requests',methods=['POST','GET'])
# def tasks():
#     global camera
#     if request.method == 'POST':
#         if request.form.get('click') == 'Capture':
#             global capture
#             capture=1
            
#     elif request.method=='GET':
#         return render_template('index.html')
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

# cv2.destroyAllWindows()
