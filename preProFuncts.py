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

def scaleImage(input_path, output_path, dim):
    image = Image.open(input_path)
    resized = image.resize(dim)
    name = os.path.basename(input_path)
    resized.save(output_path + "\\" + name)
    print(f"Scaled {input_path}: [{image.size}]->[{dim}]")


def scaleFolder(folderPath, output_path, dim):
    for root, _, files in os.walk(folderPath):
        for f in files:
            input_path = os.path.join(root, f)
            scaleImage(input_path, output_path, dim)


def imageToPx(input_path):
    image = Image.open(input_path)
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




#arr = imageToPx("C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled_sizeprocessed\\49ers.png")
#imagesPx = folderToPx("C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled_sizeprocessed")
#for i in imagesPx:
#    print(len(i))

#folderWebPtoPNG("C:\\Users\\vjgti\\Downloads\\NFL_teams", "C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled")
#scaleFolder("C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled", "C:\\Users\\vjgti\\PycharmProjects\\cs3200_termProject\\.venv\\images\\NFL_teams\\labeled_sizeprocessed", (500, 500))