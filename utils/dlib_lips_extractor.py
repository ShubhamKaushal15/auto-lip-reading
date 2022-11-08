import cv2, os, dlib
from IPython.display import Image
from imutils import face_utils
# from google.colab.patches import cv2_imshow

def show_lips(face_path, faceDetector, landmarkDetector):

  im = cv2.imread(face_path)

  dets = faceDetector(im, 1)
  shape = face_utils.shape_to_np(landmarkDetector(im, dets[0]))[48:] # get lips points
  
  # get coords
  x_min, y_min = min(shape[:, 0]), min(shape[:, 1])
  x_max, y_max = max(shape[:, 0]), max(shape[:, 1])

  padding = 5
  # padding needs to be reconsidered
  cv2.imshow(im[y_min - padding : y_max + padding, x_min - padding  : x_max + padding]) # cv2_imshow in colab

def main():
    PREDICTOR_PATH = os.path.join(os.getcwd(), "shape_predictor_68_face_landmarks.dat")

    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    show_lips(os.path.join('test_imgs', 'smile_face.jpg'), faceDetector, landmarkDetector)
    show_lips(os.path.join('test_imgs', 'test_face.png'), faceDetector, landmarkDetector)
