# 1- Convert all the images from RGB color space to LAB color space
# 2- Use the L channel as an input to the neural network
# 3- Train the nural net to predict the AB channels
# 4- Combine the L channel with the predicted AB channel
# 5- Convert the LAB image back to RGB

import numpy as np
import argparse
import cv2

# Paths to load the model
PROTOTXT = "model/colorization_deploy_v2.prototxt"
POINTS = "model/pts_in_hull.npy"
MODEL = "model/colorization_release_v2.caffemodel"

# Argparses
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Path to black & white image")
args = vars(ap.parse_args())

# Load the model
print("Loading the model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for AB channel
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load input image
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

print("Coloring the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original image", image)
cv2.imshow("Colored image", colorized)
cv2.waitKey(0)