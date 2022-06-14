# Classification-of-handwritten-numeric-data-MNIST-using-different-ML-algorithms

**Introduction:** 

Handwriting recognition (HWR) is the ability of a computer to receive and interpret
intelligible handwritten input from sources such as paper documents, photographs, touchscreens and other devices. The image of the written text may be sensed "off-line" from a piece
of paper by optical scanning (optical character recognition) or intelligent word recognition.
Alternatively, the movements of the pen tip may be sensed "on-line", for example by a pen-based
computer screen surface, a generally easier task as there are more clues available. Off-line HWR
includes anything that involves parsing a scanned document, and extracting information from it
with regards to the handwriting. The execution of this task with machine learning techniques is
comparatively involved, due to people having generally very different writing styles. Some of the
applications where a machine learning algorithm could find use within this realm are:

• Handwriting recognition for authentication of signatures

• ZIP code addresses on letters (digit recognition)

• Putting handwritten text directly in some type of digitized output etc.

Broadly speaking, the task of handwriting recognition is broken down into two parts of equal
importance. 1) Preprocessing the raw data which usually comes in scanned format(thresholding,
noise removal, line/word/letter segmentation and 2) training an ML algorithm to perform the
task. Generally preprocessing is extremely important in optimizing the performance of the
algorithm, although it looks like less careful preprocessing is necessary as the complexity and
non-linearities of the algorithm increase.

**Dataset:** To train our algorithm, we will use the data provided by the MNIST Database of
Handwritten Digits (see http://yann.lecun.com/exdb/mnist/). The dataset has some clear
advantages for beginners, since it already partitioned and preprocessed. The images have been
standardized in size and centered, so that removes the need for us to process a given scanned
document to extract single digits. The dataset consists of a training set of size~60,000 and a test
set of size ~10,000 digits respectively. The images have also been digitized in 20x20 pixel grid in
grayscale and have been centered on their center of mass.

**Algorithm architecture overview:** The goal is to use (at least) two different classification
algorithms for the digits and compare and contrast the results upon training them. One of them
has already been implemented for us in Notebook 7 (Softmax classification), but we would also
like to implement an SVM algorithm, that has not been covered in the review, to note the
differences between a classifier which is based on a generalized linear model like softmax and a
non-linear classifier like an SVM. Time permitting, we would like to attempt an implementation
that involves training a convolutional neural network with Keras, and see if we can achieve good
recognition efficiency, given that conv. nets are well known for their great performance in picture
recognition with minimal preprocessing.
