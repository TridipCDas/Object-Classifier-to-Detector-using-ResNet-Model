# Object-Classifier-to-Detector-using-ResNet-Model

The algorithm consists of the following steps:

1: Input an image
2: Construct an image pyramid
3: For each scale of the image pyramid, run a sliding window
  3a: For each stop of the sliding window, extract the ROI
  3b: Take the ROI and pass it through our CNN originally trained for image classification
  3c: Examine the probability of the top class label of the CNN, and if meets a minimum confidence, record (1) the class label and (2) the location of the sliding window
4: Apply class-wise non-maxima suppression to the bounding boxes
5: Return results to calling function

REFERENCE: https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/
