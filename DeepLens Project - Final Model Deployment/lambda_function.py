#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
import os
import json
import time
import numpy as np
import awscam
import cv2
import mo
import greengrasssdk
from utils import LocalDisplay

import sys
import onnxruntime as rt

from PIL import Image

# For nicer printing
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# Function to preprocess the image frames from the video stream
def preprocess(image):
    # Convert the image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Center crop
    midx, midy = int(grayImage.shape[1]/2), int(grayImage.shape[0]/2)
    crop_img = grayImage[:, midx-midy:midx+midy]

    # Resize with Antialias filter
    img = Image.fromarray(crop_img)
    img = img.resize((256, 256), Image.ANTIALIAS)

    # Convert back to array, divide by 255, 
    # and reshape to ONNX model reqired shape
    img = np.asarray(img, dtype=np.float32) / 255
    img = np.reshape(img, (1,256,256,1))

    return img


# Function to return the ONNX model predictions
def makeInferences(sess, input_img):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onx = sess.run([output_name], {input_name: input_img})[0]
    return pred_onx


# Needed to be an AWS Lambda function
def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    return


# Create an IoT client for sending messages to the cloud.
client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ["AWS_IOT_THING_NAME"])


# Runs infinitely on the DeepLens device
def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""    
    try:
        model_directory = "/opt/awscam/artifacts/"
        model_name = "my_model.onnx" # onnx-model

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # When the ONNX model is imported via DeepLens console, the model is copied
        # to the AWS DeepLens device, which is located in the "/opt/awscam/artifacts/".
        model_file_path = os.path.join(model_directory, model_name)
        sess = rt.InferenceSession(model_file_path)
        
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
                
            # Preprocess the frame to crop it into a square and
            # resize it to make it the same size as the model's input size.
            input_img = preprocess(frame)

    

            # Inference.
            inferences = makeInferences(sess, input_img)
            inference = np.argmax(inferences) + 1 # + 1 because of zero-indexing of classes

            # Add the label of predicted digit to the frame used by local display. 
            cv2.putText(frame, f"{inference}", (300,600), cv2.FONT_HERSHEY_SIMPLEX, 8, (255,0,0), 7)

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
 
            # Outputting the result logs as "MQTT messages" to AWS IoT.
            cloud_output = {}
            cloud_output["scores"] = inferences.tolist()
            print(inference, cloud_output)

    except Exception as ex:
        # Outputting error logs as "MQTT messages" to AWS IoT.
        print('Error in lambda {}'.format(ex))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("error details:" + str(exc_type) + str(fname) + str(exc_tb.tb_lineno))

infinite_infer_run()
