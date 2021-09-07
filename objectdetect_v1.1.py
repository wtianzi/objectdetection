# use tensorflow faster_rcnn trained model to do object detection.
# ref to: https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1
# detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1")
# for search and rescue project
# tianzi 20210907

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2



def main():
    score_thred = 0.1
    num_thred = 10

    # load model
    modelpath = "./faster_rcnn_resnet101_v1_640x640/"
    detector = tf.keras.models.load_model(modelpath)


    folder = "C:/Workspace/TensorSpace/pro1/images/test/"
    # get image list
    if not os.path.exists(folder):
        return "no images"
    for root,dnames,f_names in os.walk(folder):
        for imagepath in f_names:
            imagename=folder+"/"+imagepath
            image_tensor=GetImageTensor(imagename)
            detector_output = detector(image_tensor)

            # detection is finished, draw bounding box to image
            boundingboxes = np.array(detector_output["detection_boxes"])
            detection_scores = np.array(detector_output["detection_scores"][0])

            arr_sorted = np.sort(detection_scores)[::-1]
            if len(arr_sorted) > num_thred:
                score_thred = arr_sorted[num_thred]

            DrawBoundingBoxes(imagename, boundingboxes, detection_scores,score_thred)


    return

def DrawBoundingBoxes(imagename,boundingboxes,detection_scores,score_thred):
    image_org = cv2.imread(imagename)
    width = image_org.shape[1]
    height = image_org.shape[0]
    print(width, height)

    for i in range(0, boundingboxes.shape[1]):
        if detection_scores[i] > score_thred:
            # item : [ymin, xmin, ymax, xmax]
            print(boundingboxes[0][i],detection_scores[i])
            image_org = DrawBoundingBox(image_org, width,height, boundingboxes[0][i])
    filename = imagename.replace('.', '_101.')
    cv2.imwrite(filename, image_org)
    print(filename)
    return filename

def DrawBoundingBox(image_org,width,height,arr_box):
    # detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
    x1 = int(width * arr_box[1])
    x2 = int(width * arr_box[3])

    y1 = int(height * arr_box[0])
    y2 = int(height * arr_box[2])

    image_org=cv2.rectangle(image_org, (x1, y1), (x2, y2), (255,0,0), 1)
    return image_org


def ShowImage(image_org):
    plt.figure()
    plt.imshow(image_org)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    return

def GetImageTensor(imgname,width=600,height=600):
    img_raw = tf.io.read_file(imgname)
    img_tensor = tf.image.decode_image(img_raw)
    print(img_tensor.shape)
    print(img_tensor.dtype)
    if img_tensor.shape[2] > 3:
        # png to jpg
        img_tensor = [img_tensor[:, :, 0], img_tensor[:, :, 1], img_tensor[:, :, 2]]
        img_tensor = tf.transpose(img_tensor, perm=[2, 1, 0])
        img_tensor = tf.transpose(img_tensor, perm=[1, 0, 2])
        print("reshape:", img_tensor.shape)

    img_final = tf.image.resize(img_tensor, [width, height])
    print(img_final.shape)
    print(img_final.numpy().min())
    print(img_final.numpy().max())

    image_tensor = (np.expand_dims(img_final, 0))
    print(image_tensor.shape)
    return image_tensor

if __name__ == "__main__":
    # execute only if run as a script
    main()