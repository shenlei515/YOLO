import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import imageio
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.model.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


# def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
#     """Filters YOLO boxes by thresholding on object and class confidence.
#
#     Arguments:
#     box_confidence -- tensor of shape (19, 19, 5, 1)
#     boxes -- tensor of shape (19, 19, 5, 4)
#     box_class_probs -- tensor of shape (19, 19, 5, 80)
#     threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
#
#     Returns:
#     scores -- tensor of shape (None,), containing the class probability score for selected boxes
#     boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
#     classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
#
#     Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
#     For example, the actual output size of scores would be (10,) if there are 10 boxes.
#     """
#
#     # Step 1: Compute box scores
#     ### START CODE HERE ### (≈ 1 line)
#     box_scores = box_confidence * box_class_probs
#     ### END CODE HERE ###
#
#     # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
#     ### START CODE HERE ### (≈ 2 lines)
#     box_classes = K.argmax(box_scores, axis=-1)
#     box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
#     ### END CODE HERE ###
#
#     # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
#     # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
#     ### START CODE HERE ### (≈ 1 line)
#     filtering_mask = box_class_scores >= threshold
#     ### END CODE HERE ###
#
#     # Step 4: Apply the mask to scores, boxes and classes
#     ### START CODE HERE ### (≈ 3 lines)
#     scores = tf.boolean_mask(box_class_scores, filtering_mask)
#     boxes = tf.boolean_mask(boxes, filtering_mask)
#     classes = tf.boolean_mask(box_classes, filtering_mask)
#     ### END CODE HERE ###
#
#     return scores, boxes, classes


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_class_conf=tf.multiply(box_confidence,box_class_probs)
    box_class_max=K.max(box_class_conf,axis=-1)
    box_class_max_index=K.argmax(box_class_conf,axis=-1)
    box_class_filted=tf.boolean_mask(box_class_max,box_class_max>=0.6)
    box_class_filted_index=tf.boolean_mask(box_class_max_index,box_class_max>=0.6)
    box_class_filted_box=tf.boolean_mask(boxes,box_class_max>=0.6)
    return box_class_filted,box_class_filted_box,box_class_filted_index

def iou(box1,box2):
    x1i=max(box1[0],box2[0])
    y1i=max(box1[1],box2[1])
    x2i=min(box1[2],box2[2])
    y2i=min(box1[3],box2[3])
    s_1=abs(box1[0]-box1[2])*abs(box1[1]-box1[3])
    s_2=abs(box2[0]-box2[2])*abs(box2[1]-box2[3])
    s_lap=max(x1i-x2i,0)*max(y1i-y2i,0)
    return s_lap/(s_2+s_1-s_lap)

def yolo_non_max_suppression(scores,boxes,classes,max_boxes=10,iou_threshold=0.5):
    box_list_score=[]
    box_list_class=[]
    box_list_boxes=[]
    i=0
    while(i<max_boxes and len(boxes)!=0):
        main_box=K.argmax(scores)
        box_list_score.append(scores[main_box])
        box_list_class.append(classes[main_box])
        box_list_boxes.append(boxes[main_box])
        # for t in range(i+1,len(boxes)):不能在for...in...里面用变化的范围
        t=0
        while(t<=len(boxes)-1):#而while可以更新
            if(t==main_box):
                t=t+1
                continue
            if (iou(boxes[t],boxes[main_box])>iou_threshold):
                scores=np.delete(scores,t,axis=0)
                classes=np.delete(classes,t,axis=0)
                boxes=np.delete(boxes,t,axis=0)
                t=t-1
                if(main_box>t):
                    main_box=main_box-1
            t=t+1
        t=0
        scores=np.delete(scores,main_box,axis=0)
        classes=np.delete(classes,main_box,axis=0)
        boxes=np.delete(boxes,main_box,axis=0)
        i=i+1
    return box_list_score,box_list_boxes,box_list_class


def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy,box_wh)
    scores,boxes,classes=yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)

    boxes=scale_boxes(boxes,image_shape)
    scores,boxes,classes=yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)

    return scores,boxes,classes


class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
image, image_data = preprocess_image("images/" + "test.jpg", model_image_size = (608, 608))
yolo_model = load_model("model_data/yolo.h5")
outputs = tf.convert_to_tensor(yolo_model.predict(image_data))
yolo_outputs = yolo_head(outputs, anchors, len(class_names))
out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape)

print('Found {} boxes for {}'.format(len(out_boxes), "test.jpg"))
# Generate colors for drawing bounding boxes.
colors = generate_colors(class_names)
# Draw bounding boxes on the image file
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
# Save the predicted bounding box on the image
image.save(os.path.join("out", "test.jpg"), quality=90)
# Display the results in the notebook
output_image = imageio.imread(os.path.join("out", "test.jpg"))
imshow(output_image)
plt.show()