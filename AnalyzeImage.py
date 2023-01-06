import requests
import httplib2
import PIL.Image as Image
import io
import cv2
from datetime import date
import datetime
import time
import torch
import numpy as np

def readAndStoreImage():

    # Replace driveway.lan with your camera domain url or ip address    

    cameraImageUrl =  "http://driveway.lan/cgi-bin/snapshot.cgi?1" 
    h = httplib2.Http(".cache")
    h.add_credentials('user', 'password')  # Basic authentication   
    resp, content = h.request(cameraImageUrl, "GET")
    return content

    
def imageAnalysis(content):

    #This code is for darwing bounding lines once image has been detected

    blue = (255, 0, 0)
    white= (255, 255, 255)

    _imageInference = False

    classes = []    

    # Read the byte array connect 

    img = Image.open(io.BytesIO(content))

    model.classes = [0,2]

    # Pass the image to the model 

    results = model(img, size=640)

    data = results.pandas().xyxy[0].to_dict(orient='records')

    if len(data) > 0:

        _imageInference = True

        nparr = np.fromstring(content, np.uint8)
        img_np = cv2.imdecode(nparr, -1)

        for items in data:
            cv2.rectangle(img_np, (int(items['xmin']), int(items['ymin'])), (int(items['xmax']), int(items['ymax'])), blue, 2)
            className = items['name']
            classes.append(className)
            cv2.putText(img_np, className, (int(items['xmin']), int(
                items['ymax']) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)
        
        is_success, im_buf_arr = cv2.imencode(".jpg", img_np)

        return _imageInference, im_buf_arr.tobytes(), classes

    return _imageInference, None, None

if __name__ == '__main__':

    # Load the model 
    # The pretrained model of Yolo v5 can identify 79 objects. https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    content = readAndStoreImage()
    _inferedImageFlag, _inferredImageByteArray, classes = imageAnalysis(content)

    if _inferedImageFlag:
        image = Image.open(io.BytesIO(_inferredImageByteArray))
        image.save("inferredimage.jpg")




    

    