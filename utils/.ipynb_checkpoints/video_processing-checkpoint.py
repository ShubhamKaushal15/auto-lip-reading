import numpy as np
import os
from scipy import ndimage
import dlib
# from scipy.misc import imresize
import cv2
import skvideo.io
from imutils import face_utils

class Video(object):
    def __init__(self, vtype='mouth', detector=None, predictor=None):
        if vtype == 'face' and (predictor is None or detector is None):
            raise AttributeError('Face video needs to be accompanied face predictor AND detector')
        self.detector = detector
        self.predictor = predictor
        self.vtype = vtype

    def from_frames(self, path):
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        self.handle_type(frames)
        return self

    def from_video(self, path):
        frames = self.get_video_frames(path)
        self.handle_type(frames)
        return self

    def from_array(self, frames):
        self.handle_type(frames)
        return self

    def handle_type(self, frames):
        if self.vtype == 'mouth':
            self.process_frames_mouth(frames)
        elif self.vtype == 'face':
            self.process_frames_face(frames)
        else:
            raise AttributeError('Video type should be face or mouth')

    def process_frames_face(self, frames):
        mouth_frames = self.get_frames_mouth(frames)
        self.face = np.array(frames)
        self.mouth = np.array(mouth_frames)
        self.set_data(mouth_frames)

    def process_frames_mouth(self, frames):
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        self.set_data(frames)

    def get_frames_mouth(self, frames):
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        PADDING = 0.2
        mouth_frames = []
        for frame in frames:
            dets = self.detector(frame, 1)
            np_mouth_points = face_utils.shape_to_np(self.predictor(frame, dets[0]))[48:] # get lips points

            x_min, y_min = min(np_mouth_points[:, 0]), min(np_mouth_points[:, 1])
            x_max, y_max = max(np_mouth_points[:, 0]), max(np_mouth_points[:, 1])
            padding_y = int((y_max - y_min) * PADDING)
            padding_x = int((x_max - x_min) * PADDING)

            crop = frame[y_min - padding_y: y_max + padding_y, x_min - padding_x: x_max + padding_x]

            mouth_frames.append(cv2.resize(crop, (MOUTH_WIDTH, MOUTH_HEIGHT)))
        return mouth_frames

    def get_video_frames(self, path):
        videogen = skvideo.io.vreader(path)
        frames = np.array([frame for frame in videogen])
        return frames

    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0,1) # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0,2).swapaxes(0,1) # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames) # T x W x H x C
        # if K.image_data_format() == 'channels_first':
        #     data_frames = np.rollaxis(data_frames, 3) # C x T x W x H
        self.data = data_frames
        self.length = frames_n