import sys
sys.path.append("./packages")
import os
import numpy as np
import boto3
import onnxruntime as rt
from PIL import Image

# Function to crop and resize to the same specifications as the model was trained on
# and returning an appropriate numpy array
def preprocess(image):
    midx, midy = int(image.shape[1]/2), int(image.shape[0]/2)
    crop_img = image[:, midx-midy:midx+midy]
    img = Image.fromarray(crop_img)
    img = img.resize((256, 256), Image.ANTIALIAS)
    processed_image = np.asarray(img, dtype = np.float32)
    processed_image = np.asarray(processed_image, dtype=np.float32) / 255
    processed_image = np.reshape(processed_image, (1,256,256,1))
    return processed_image


# ONNX model inference function
def makeInference(sess, input_img):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onx = sess.run([output_name], {input_name: input_img})[0]
    return pred_onx


# Lambda handler for AWS lambda function 
def lambda_handler(event, context):
    # Important definitions
    s3_bucket_name = "heroku-deployment"
    lambda_tmp_directory = "/tmp"
    model_file_name = "my_model.onnx"
    input_file_name = "digit.jpg"
    output_file_name = "results.txt"

    # Making probability print-out look pretty.
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    try:
        # Download test image and model from S3.
        client = boto3.client('s3')
        client.download_file(s3_bucket_name, input_file_name, 
                                os.path.join(lambda_tmp_directory, input_file_name))                     
        client.download_file(s3_bucket_name, model_file_name, 
                                os.path.join(lambda_tmp_directory, model_file_name))
    except:
        print("Couldn't properly download the files from s3")

    # Import input image in grayscale and preprocess it.
    image = np.asarray(Image.open(os.path.join(lambda_tmp_directory, 
                        input_file_name)).convert("L"), dtype=np.float32)
    processed_image = preprocess(image)

    # Make inference using the ONNX model.
    sess = rt.InferenceSession(os.path.join(lambda_tmp_directory, model_file_name))
    inferences = makeInference(sess, processed_image)
    
    print("predicted on image")
    print(f"predicted {np.argmax(inferences)}")

    # Output probabilities in an output file.
    f = open(os.path.join(lambda_tmp_directory, output_file_name), "w+")
    f.write("Predicted: \"%d\" " % (np.argmax(inferences)+1))
    f.write(" Probability: %f" % inferences.max())
    f.close()

    try:
        # Upload the output file to the S3 bucket.
        client.upload_file(os.path.join(lambda_tmp_directory, 
                            output_file_name), s3_bucket_name, output_file_name)
    except:
        print("Couldn't upload the file..........")