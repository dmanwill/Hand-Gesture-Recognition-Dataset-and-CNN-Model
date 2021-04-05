import os
# GUI Packages.
import matplotlib.pyplot as plt
import ipywidgets as widgets
import time

# AWS Packages.
import boto3


# AWS Variables.
accessKeyID = os.environ["AWS_ACCESS_KEY_ID"]
secretAccessKey = os.environ["AWS_SECRET_ACCESS_KEY"]
s3BucketName = "heroku-deployment"
lambda_function_name = "heroku_deployment"
inputImageFileName = "digit.jpg"
resultsDataFileName = "results.txt"


def parseAndShowResults(resultsDataFileName):

    with open(resultsDataFileName, "r") as results:
        # Extract prediction results.
        # Find the prediction value with the highest prediction value.
        print(open(resultsDataFileName).read())
      
        # Display predicted value, prediction probability, and image of the hand-writtent digit that was classified.
        display(widgets.Image(value=imageBytesData))
        
        pass

  

## AWS Image Upload callback function and button ##

# Upload digit.png to S3 to produce the results.txt using lambda.
def awsImageUpload(data):
    client = boto3.client(
        's3',
        aws_access_key_id=accessKeyID,
        aws_secret_access_key=secretAccessKey
    )
    
    # Upload digit.png to S3.
    try:
        client.upload_file(inputImageFileName, s3BucketName, inputImageFileName)
        print("Upload Successful")
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

    try:
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        lambda_client.invoke(FunctionName=lambda_function_name, InvocationType='Event')
        print("AWS Processing...")
    except:
        print("Couldn't properly call AWS Lambda function")
    
    # Waiting and checking to see if the results.txt has been produced and placed in S3 from Lambda.
    time.sleep(awsProgressRefreshRateSlider.value)
    fount_text = False
    while(not fount_text):
        time.sleep(awsProgressRefreshRateSlider.value)
        try:
            client.download_file(s3BucketName, resultsDataFileName, resultsDataFileName)
            fount_text = True
        except:
            print("waiting for result")
            
     
    # Removing input digit.jpg and output results.txt from S3.
    client.delete_object(Bucket=s3BucketName, Key = inputImageFileName)
    client.delete_object(Bucket=s3BucketName, Key = resultsDataFileName)

    
    # Display Results
    parseAndShowResults(resultsDataFileName)



## Image upload callback function and button ##

def selectimage2upload(imageData):
    # Due to the file structure, image file name needs to be
    # extracted to access the bytes data of the image.
    imageFileName = list(imageData["new"].keys())[0]
    
    # Image bytes data.
    global imageBytesData
    imageBytesData = imageData["new"][imageFileName]["content"]
    
    # Writing image file to current directory with "inputImageFileName".
    with open(inputImageFileName, "wb") as imageFile:
        imageFile.write(imageBytesData)
    
    # Displaying uploaded image in GUI.
    display(widgets.Image(value=imageBytesData))
    
    # Showing AWS GUI Components after image is uploaded.
    display(awsProgressRefreshRateSlider)
    display(awsUploadButton)
    
    awsUploadButton.on_click(awsImageUpload)



def createDashBoard():
    # Allows the buttons to be accessed globally: Necessary
    # since some callback functions are dependent on these
    # widgets.
    global awsUploadButton
    global awsProgressRefreshRateSlider
    global image_upload_button
    
    awsUploadButton = widgets.Button(description='Upload to AWS')
    
    # AWS Image Upload Button.
    image_upload_button = widgets.FileUpload()
    
    # AWS Progress Refresh Rate Selector.
    awsProgressRefreshRateSlider = widgets.FloatSlider(max = 1.0)
    
    # Display GUI.
    display(image_upload_button)
    
    time.sleep(0.1)
    
    def when_loaded(change):
        selectimage2upload(change)
        
    image_upload_button.observe(when_loaded, names='value')