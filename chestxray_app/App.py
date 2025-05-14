#Import libraries
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import csv
from typing import List
import io
import uvicorn #type: ignore
from fastapi import FastAPI, File, UploadFile #type: ignore
from fastapi.responses import JSONResponse #type: ignore
from fastapi.middleware.cors import CORSMiddleware #type: ignore
from starlette.responses import StreamingResponse #type: ignore
#Initialize FastAPI app
app = FastAPI()
#Allow front end hosted at port 3000 to access api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#Load model
model = tf.keras.models.load_model('Final_DenseNet_ChestXRay.keras')
#Initialize labels used by model
labels = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration','Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
#Preprocess image used by model
def preprocess_image(image_data):
    #Convert binary information sent by front end to pil image, then resize image to 224x224 and convert to RGB for model
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    #Convert image to float32 numpy array, add batch dimension, and preprocess for DenseNet121
    image_array = np.array(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.densenet.preprocess_input(image_array)
    return image_array
#Get images from upload file object and process it through the model 
@app.post("/process-images")
async def process_images(images: List[UploadFile] = File(...)):
    #Initialize results array
    results = []

    try:
        #Loop over each image in list
        for image in images:
            #Get bytes from file upload
            image_data = await image.read()
            #Get image from bytes
            preprocessed_image = preprocess_image(image_data)
            #Use image to make prediction
            predictions = model.predict(preprocessed_image)[0]
            ##Get index and label of class with highest prediction score
            predicted_index = int(np.argmax(predictions))
            predicted_label = labels[predicted_index]
            #Get confidence score for prediction
            confidence_score = float(predictions[predicted_index]) * 100
            #Append results to results array
            results.append([image.filename, predicted_label, f"{confidence_score:.2f}%"])
        
        #Create file-like object to store csv information
        output = io.StringIO()
        #Create csv writer object to write csv information
        writer = csv.writer(output)
        #Write column names
        writer.writerow(["image_filename", "prediction", "confidence"])
        #Write data entries for each row
        writer.writerows(results)
        #Reset file pointer to beginning of csv string to allow for reading csv information from the beginning 
        output.seek(0)

        #Return csv as downloadable file
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    #Send back error message if try block fails
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
#Start FastAPI app on localhost port 5000
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
