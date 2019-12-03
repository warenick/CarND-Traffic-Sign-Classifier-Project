import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_imgs(imgs):
    for img in imgs:
        plt.figure()
        plt.imshow(img)

def show_imgs_together(imgs,titles):
    fig = plt.figure(figsize=(len(imgs)/2, len(imgs)*2))
    for i in range(len(imgs)):
        fig.add_subplot(len(imgs)/4+1, 4, i+1)
        plt.title(titles[i])
        plt.imshow(imgs[i])
    # plt.savefig("./images/datasetImgs2.png")

def show_imgs_together2(imgs,titles):
    fig = plt.figure(figsize=(5, 2))
    for i in range(len(imgs)):
        fig.add_subplot(len(imgs)/4+1, 4, i+1)
        plt.title(titles[i])
        plt.imshow(imgs[i])
    # plt.savefig("./images/new_imgs.png")

def show2imgs(img1,img2,name="image.png"):
    fig = plt.figure(figsize=(10,20))
    fig.add_subplot(1, 2, 1)
    plt.title("img1")
    plt.imshow(img1)
    fig.add_subplot(1, 2, 2)
    plt.title("img2")
    plt.imshow(img2)

    # plt.savefig("./images/"+name)


def show_img(img):
    plt.figure()
    plt.imshow(img)


