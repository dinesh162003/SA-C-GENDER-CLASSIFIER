# SA-C-GENDER-CLASSIFIER
# Algorithm
1. To classify the gender of a person use the DeepFace library.
2. DeepFace library is developed based on deep learning algorithms.
3. Import the deepface class from DeepFace library, cv2 class from openCv2 library and Matplot library according to the requirements.
4. Load and display the imported image.
5. Pass the image to DeepFace library and analyze the image to predict gender of a person.
6. This prediction is stored in result variable.
7. Finally print the prediction using this algorithm.

## Program:
```
/*
Program to implement 
Developed by   : Dinesh.S
RegisterNumber :  212220230011
*/
```

```
!pip install deepface
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
img1=cv2.imread('samantha.jpg')
plt.imshow(img1[:,:,::-1])
plt.show()
result=DeepFace.analyze(img1,actions=['gender'])
print("Gender : ",result['gender'])
img2=cv2.imread('hendry cavill.jpg')
plt.imshow(img1[:,:,::-1])
plt.show()
result=DeepFace.analyze(img2,actions=['gender'])
print("Gender : ",result['gender'])
```

## OUTPUT:
![image](https://user-images.githubusercontent.com/75235159/174522461-d19a2215-31cd-4955-a816-82bae430bcea.png)


![image](https://user-images.githubusercontent.com/75235159/174522364-c5120524-0915-45a3-a11b-1cddd11bcbe5.png)
![image](https://user-images.githubusercontent.com/75235159/174522503-4b5ec2f8-650f-4031-ba4d-0535f9d83f4f.png)

![image](https://user-images.githubusercontent.com/75235159/174522389-cb1dadeb-0579-4e5a-a427-23ce3a5feb31.png)



2. DEMO VIDEO YOUTUBE LINK:

https://youtu.be/PYGgniwel2c
