# Perception Challenge
![image](https://user-images.githubusercontent.com/85666623/192154907-28a95447-cd3f-4f2e-9571-251f5614a714.png)

Methodolgy: First, read and resize the picture. Then store the picture in tubes with the order of B,G,R. Transfer RGB to HSV and prodcue a summed up mask with two binary image masks. Since the color differnece is big, we can use binary image to better extract the edge of target points. Then use function cv2.moments to capture features of moments of red items. Sort all points we get based on Y axis to find 4 red items (top 2 and bottom 2). And we do some simple work, sort all points by the distance between them and the middle points and distinguish left points and right points, then draw left line and right line.

What did you try and why do you think it did not work: I spent a lot of time on searching tutorial about RGB, HSV, image moments and other opencv stuff. For the first try, I tried to simply extract all red parts of the image, then recognize and combine red points on two lines. But there are two major interference misleading the machine when recognizing all red parts since these interferences are also red (doors and "EXIT" sign). Then I learned about cv2.moments to capture the features of moments and find cloest 4 points (top 2 and bottom 2) to make lines to deal with this. This is the most difficult part for me. Then just sort all points by distance, distinguish left and right and draw two lines.

What libraries are used: 1. import cv2 2. import numpy as np
