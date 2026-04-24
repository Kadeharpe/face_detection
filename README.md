# Face Recognition Project

Dataset:
- CelebA (not included due to size)

How to run:
1. Go to CelebA img and download img_align_celeba and anno and download identity_celeba.txt
2. Place in project folder
3. Run pip install -r requirements.txt to install all required files to run
4. Run train_yolo_cls.py
    
After running:

The dataset (celeba_yolo_cls) is automatically generated.

If you change dataset parameters such as:
- number of identities
- number of images per identity

You must delete the existing "celeba_yolo_cls" folder before rerunning the script so the dataset can be rebuilt correctly.