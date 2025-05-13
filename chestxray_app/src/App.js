//Import libraries
import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

function App() {
  //Create state variables
  const [selectedImageFiles, setSelectedImageFiles] = useState([]);
  const [previewImageURLs, setPreviewImageURLs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isProcessingComplete, setIsProcessingComplete] = useState(false);
  //Set selectedImageFiles to list of user selected image files, set setPreviewImageURLs to first 12 user selected image files, and set isProcessingComplete to false
  const onDrop = (acceptedFiles) => {
    setSelectedImageFiles(acceptedFiles);
    const previews = acceptedFiles.slice(0, 12).map(file => URL.createObjectURL(file));
    setPreviewImageURLs(previews);
    setIsProcessingComplete(false);
  };
  //Set drop zone that can be selected to upload files, allow it to accept mutiple images and multiple image file types
  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: 'image/jpeg, image/png, image/jpg',
    multiple: true
  });
  //Execute when Upload Images button is seleceted
  const handleUpload = async () => {
    //If no images were selected alert user
    if (selectedImageFiles.length === 0) {
      alert('Please select images to upload:');
      return;
    }
    //Set formData object to a list of key-value pairs where each key is the string 'images' and the values are the user selected images
    const formData = new FormData();
    selectedImageFiles.forEach((file) => {
      formData.append('images', file);
    });
    //Set isProcessing to true and set isProcessingComplete to false
    setIsProcessing(true);
    setIsProcessingComplete(false);

    try {
      //Upload images to api endpoint as binary data and expect the api to return binary data
      const response = await axios.post('http://localhost:5000/process-images', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
      });
      //Get binary data from api response and indicate the data represents a csv file
      const blob = new Blob([response.data], { type: 'text/csv' });
      //Set url to blob that can be used for downloading
      const url = window.URL.createObjectURL(blob);
      //Create anchor tag and set its destination link to the blob url
      const link = document.createElement('a');
      link.href = url;
      //Set download attribute and set the default file name to predictions.csv
      link.setAttribute('download', 'predictions.csv');
      //Append link to the html dom to enable link.click(); to work 
      document.body.appendChild(link);
      //Simulate a click event to trigger the download
      link.click();
      //Remove link from the html dom after download is triggered
      link.remove();
      //Set isProcessing to false and the isProcessingComplete to true
      setIsProcessing(false);
      setIsProcessingComplete(true);
    //Display error message if upload fails
    } catch (error) {
      console.error('Error uploading images:', error);
    }
  };

  return (
    <div>
      {/*Use getRootProps to define the dropzone, create an input element for file selection, and define stylistic features for dropzone*/}
      <div {...getRootProps()} style={{ border: '2px dashed #ccc', padding: '20px', textAlign: 'center' }}>
        <input {...getInputProps()} />
        <p>Click to select images</p>
      </div>
      {/*Use previewImageURLs to create up to 12 100x100 image previews, define stylistic features for previews, and use selectedImageFiles to display number of selected images*/}
      {previewImageURLs.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '20px' }}>
          {previewImageURLs.map((url, idx) => (
            <img key={idx} src={url} alt={`Preview ${idx}`} style={{ width: '100px', height: '100px', objectFit: 'cover' }} />
          ))}
          <p style={{ marginTop: '10px' }}>{selectedImageFiles.length} images selected</p>
        </div>
      )}
      {/*Define upload images button, disable button when processing images and change button text to indicate image processing*/}
      <button
        onClick={handleUpload}
        style={{ marginTop: '20px' }}
        disabled={isProcessing}
      >
        {isProcessing ? 'Processing Images' : 'Upload Images'}
      </button>
      {/*Display additional text to indicate when images are processing and when processing is complete*/}
      {isProcessing && <p>Processing Images</p>}
      {isProcessingComplete && <p>Processing Complete</p>}
    </div>
  );
}

export default App;
