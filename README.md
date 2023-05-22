**Face Detection using Computer Vision**
This repository contains a Python script that uses computer vision techniques to detect faces in an image. The script utilizes the OpenCV library, which provides various tools and algorithms for computer vision tasks.

**Prerequisites**
1. Python 3.x
2. OpenCV library (install using pip install opencv-python)
**Usage**
Clone the repository or download the face_detection.py file.

Place the image you want to perform face detection on in the same directory as the script. Alternatively, you can specify the path to the image in the image_path variable within the script.

Run the script using the following command:


_python face_detection.py_

This will open a window displaying the image with rectangles drawn around the detected faces.

Press any key to close the window and exit the script.

**Customization**
If you want to use a different image for face detection, update the image_path variable within the script with the path to your image.

You can adjust the parameters of the detectMultiScale function in the script to fine-tune the face detection algorithm. The scaleFactor parameter controls the image scale reduction, the minNeighbors parameter determines the minimum number of neighbors for each candidate rectangle, and the minSize parameter sets the minimum size of the detected face.



**Acknowledgements**
The face detection algorithm in this script uses the pre-trained Haar cascade classifier from OpenCV, which is based on the work by Viola and Jones. The Haar cascade XML file used in this script (haarcascade_frontalface_default.xml) is provided by OpenCV.

**Contributing**
Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or a pull request.

