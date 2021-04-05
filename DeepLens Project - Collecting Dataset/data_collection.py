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
import boto3


# For nicer printing
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# Function to preprocess the images as they are captured before
# uploading to the S3 bucket
def preprocess(image):
    # Convert the image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Center crop
    midx, midy = int(grayImage.shape[1]/2), int(grayImage.shape[0]/2)
    crop_img = grayImage[:, midx-midy:midx+midy]

    # Resize with Antialias filter
    img = Image.fromarray(crop_img)
    img = img.resize((256, 256), Image.ANTIALIAS)

    return img


# Needed to be an AWS Lambda function
def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    return


# Create an IoT client for sending to messages to the cloud.
client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ["AWS_IOT_THING_NAME"])


# Runs infinitely on the DeepLens device (only runs the loop to do anything for the
# specified number of times in the range of the for loop)
def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""
    try:
        s3BucketName = "dmanwill-project-dataset"      
        deepLensTempDirectory = "/tmp"

        s3Client = boto3.client(
            's3',
            aws_access_key_id="", # TODO
            aws_secret_access_key="" # TODO
        )

        local_display = LocalDisplay('480p')
        local_display.start()
        
        # Collects 200 images (can change to whatever number of images desired)
        for i in range(200):
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
                
            try:                
                # Preprocessing the image
                preprocessedImage = preprocess(frame)        
                local_display.set_frame_data(frame)

                frameImageFileName = f"image{i}.jpg"
                frameImageDeepLensLocation = os.path.join(deepLensTempDirectory, 
                                                            frameImageFileName)

                preprocessedImage.save(frameImageDeepLensLocation)
                s3Client.upload_file(Filename=frameImageDeepLensLocation, 
                                        Bucket=s3BucketName, Key=frameImageFileName)
                os.remove(frameImageDeepLensLocation)
                time.sleep(0.5)
                print(f"Saving image {i}")

            except Exception as e:
                print(e)

    except Exception as ex:
        print('Error in lambda {}'.format(ex))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("error details:" + str(exc_type) + str(fname) + str(exc_tb.tb_lineno))

infinite_infer_run()
