import os

from PIL import Image

import numpy as np

def convertWebPtoPNG(input_path, output_path):
    try:
        image = Image.open(input_path)
        image.save(output_path, "PNG")
        print(f"Converted {input_path} to PNG [{image.size}]")
    except Exception as e:
        print(f"Error on {input_path}: {e}")



def folderWebPtoPNG(folderPath, output_path):
    for root, _, files in os.walk(folderPath):
        for f in files:
            if f.endswith(".webp"):
                input_path = os.path.join(root, f)
                convertWebPtoPNG(input_path, output_path+"\\"+f.split(".")[0]+".png")


def convertWebPtoPNG(input_path, output_path):
    try:
        image = Image.open(input_path)
        image.save(output_path, "PNG")
        print(f"Converted {input_path} to PNG [{image.size}]")
    except Exception as e:
        print(f"Error on {input_path}: {e}")



counter = 0
def foldertoPNG(folderPath, output_path):
    global counter
    for root, _, files in os.walk(folderPath):
        for f in files:
            input_path = os.path.join(root, f)
            convertWebPtoPNG(input_path, output_path+"\\"+str(counter)+".png")
            counter+=1



#only provide directory path for the output path and gets file name from original file name
def scaleImage(input_path, output_path, dim):
    image = Image.open(input_path)
    resized = image.resize(dim)
    name = os.path.basename(input_path)
    resized.save(output_path + "\\" + name)
    print(f"Scaled {input_path}: [{image.size}]->[{dim}]")

#provide both the input path w/ file name and output path w/ file name
def scaleImage2(input_path, output_path, dim):
    image = Image.open(input_path)
    resized = image.resize(dim)
    resized.save(output_path)
    print(f"Scaled {input_path}: [{image.size}]->[{dim}]")

def returnScaledImageIn1DArray(input_path, dim):
    image = Image.open(input_path).convert("RGB")
    resized = image.resize(dim)
    twoD = np.array(resized)
    oneD = twoD.ravel()
    return oneD

def scaleFolder(folderPath, output_path, dim):
    for root, _, files in os.walk(folderPath):
        for f in files:
            input_path = os.path.join(root, f)
            scaleImage(input_path, output_path, dim)


def imageToPx(input_path, mode="RGB"):
    image = Image.open(input_path).convert(mode)
    twoDArray = np.array(image)
    oneDArray = twoDArray.ravel()
    #print(twoDArray[0])
    #print(oneDArray[0])
    return oneDArray

def folderToPx(folderPath):
    images = []
    for root, _, files in os.walk(folderPath):
        for f in files:
            input_path = os.path.join(root, f)
            images.append((imageToPx(input_path), f))
    return images



#borrowed from my CS4420 class simulation group project (https://github.com/vjgtigers/traffic-sim/blob/master/utility.py)
def get_unique_filename(filepath):
    """
    If filepath exists, append a number to make it unique.
    Example: file.txt -> file_1.txt -> file_2.txt, etc.
    """
    if not os.path.exists(filepath):
        return filepath

    # Split into base name and extension
    base, ext = os.path.splitext(filepath)

    counter = 1
    while True:
        new_filepath = f"{base}_{counter}{ext}"
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1




#arr = imageToPx("C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled_sizeprocessed\\49ers.png")
#imagesPx = folderToPx("C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled_sizeprocessed")
#for i in imagesPx:
#    print(len(i))

#foldertoPNG("C:\\Users\\vjgti\\Downloads\\NBA_unlabeled\\allData", "C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NBA_teams\\unlabeled")
#scaleFolder("C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NBA_teams\\unlabeled", "C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NBA_teams\\unlabeled_sizeprocessed", (500, 500))