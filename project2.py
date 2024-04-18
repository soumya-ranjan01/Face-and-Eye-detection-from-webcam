from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def frame_gen():
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            face_dector=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            eye_dector=cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
            face=face_dector.detectMultiScale(frame,1.1,7)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            for (x,y,w,h) in face:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=frame[y:y+h,x:x+w]

                eyes=eye_dector.detectMultiScale(roi_gray,1.1,3)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)


            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
              b'Content-type:image/jpeg\r\n\r\n'+frame+b'r\n')


@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/livedetect')
def livedetect():
    return Response(frame_gen(),mimetype='multipart/x-mixed-replace;boundary=frame')


if __name__=='__main__':
    app.run(debug=True)