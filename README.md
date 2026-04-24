# Face Recognition Project

Dataset:
- CelebA (not included due to size)

Goal:
The goal of this project is to build a two-stage facial recognition project that detects and classifies individuals faces.

Models used:
    Baseline:
K-nearest neighbor(kNN)
Support vector machine(SVM)
    Deep Learning:
YOLOv8 classification model

Notes:
Model will only classify identties who are already trained | 
Unknown faces will come out as unkown | 
Could possibly add to training data to further advance the model | 
test.jpg is a picture of id 9040 that has not been used in the test, train, or validate cycles so that is why it is the test 

How to run:
1. Go to CelebA img and download img_align_celeba and anno to download identity_celeba.txt which puts the identities to each photo.
2. Place in project folder
3. Run pip install -r requirements.txt to install all required files to run
4. Run train_yolo_cls.py (See *After running)
5. Run baseline_models.py
6. Test face detection and run face_detect_test.py
7. Run detect_and_classify.py and get results
    
*After running:
The dataset (celeba_yolo_cls) is automatically generated.
    If you change dataset parameters such as:
- number of identities
- number of images per identity
    You must delete the existing "celeba_yolo_cls" folder before rerunning the script so the dataset can be rebuilt correctly.

Results:
kNN | Baseline model with lowest performance
SVM | Improved baseline performance
YOLOv8 | Best performing model

Best YOLOv8 model:
    10 identities | 
    15 epochs | 
    70% Top 1 Accuracy | 
    100% Top 5 Accuracy | 

Pipline:
Imput image > Detect face > Crop image > Classify with YOLO > Predict identity