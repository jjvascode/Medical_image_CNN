# Detecting Medical Conditions in X-Ray scans of Medical Patients using CNNs
The goal of this project is to create, train, and deploy a model that can detect whether or not a given X-Ray image contains the signs of medical conditions that need further attention from a medical professional. This model will also provide a probable diagnosis for the patient based on the type of abnormalities detected in the X-Ray scans.

## Setup 

Witin the project, there are two distinct portions, the models/training/evaluation as well as the Detection App built using JavaScript. Both will have different methods of setting up that will be explained below: 

First, regardless of which portion you wish to recreate, you must first clone the repo by using the following command:
```
git clone https://github.com/jjvascode/Medical_image_CNN.git
```

### Models/Training/Evaluation

1. Once the github has been cloned, you will be able to access the files. From here you will be able to play around with the models that have been created in pursuit of this project. In order to do this, first create a virtual environment and install the needed dependencies by using the following command: 
```
python -m venv venv

# Activate (WIndows)
.\venv\Scripts\activate

# Activate (Mac)
source venv/bin/activate

pip install -r requirements.txt
```

2. From here you will be able to manipulate the models by adjusting hyperparameters as well as being able to alter model architecture. Within these models, there are a few key differences that must be considered. In models foudn within the src folder such as ```binary.py, datahandler.py, test.py, train.py``` these will need for filepaths to be added to the ```.env```. While modles/files within the ```james_model/``` and ```binary_model/``` folder will expect for the filepaths to be hard-coded into the actual files. 

3. Once these changes are made and the models and filepaths are all seutup properly, they can be run as is by executing the python file. 

### Detection App

Within the directory, you will want to split the terminal so that you can run the backend along with the React frontend. 

1. Navigate into the project directory: 
```
cd your-repo-name
```

2. Install Dependencies:
```
npm install
```

3. Start the development server: 
```
npm start
```

4. Access the app by opening a browser and going to 
```
http://localhost:3000
```
One the app is running on local host, you will then need to ensure that the backend is running as well. This can be done by using the following steps. 

1. From within the directory, you will have the option of running two different Python backends. The first of these is ```App.py``` and the other is ```Binary_App.py```. This can be done as such:
```
# To run App
python App.py

# To run Binary_app
python Binary_App.py
```

2. Only one of these needs to be run, which one is used is up to the discretion of the user. ```App.py``` will allow for the React app to perform multi-class classifications (0-14) on the medical images that are uploaded for processing. While running ```Binary_App.py``` will allow for the app to do both multi-class classification as well as providing it with the ability to also perform binary classification (normal vs abnormal) on the images.