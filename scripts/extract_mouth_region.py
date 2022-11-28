import os, fnmatch, sys, errno
from skimage import io
from video_processing import Video
import dlib
from os.path import expanduser

home = expanduser("~")

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def main(source_path, target_path, face_predictor_path, source_extension = "*.mpg"):
    predictor = dlib.shape_predictor(face_predictor_path)
    detector = dlib.get_frontal_face_detector()
    for filepath in find_files(source_path, source_extension):
        print(f"Processing: {filepath}")
        
        filepath_wo_ext = os.path.splitext(filepath)[0]
        target_dir = os.path.join(target_path, *filepath_wo_ext.split(os.path.sep)[-2:])
        
        if not os.path.exists(target_dir):
        
            video = Video(vtype='face', detector=detector, predictor=predictor).from_video(filepath)
            mkdir_p(target_dir)

            for i, frame in enumerate(video.mouth):
                io.imsave(os.path.join(target_dir, f"mouth_{i:03d}.png"), frame)

if __name__ == '__main__':
    """
    Run this script from auto-lip-reading directory

    source_path: directory of video files
    target_path: directory where processed video should be saved
    face_predictor_path: path to shape_predictor_68_face_landmarks.dat (including)
    source_extension: *.mpg or *.avi, the extension of video files to be converted
    """
    
    source_path = os.path.join(home, "grid_videos") #sys.argv[1]
    target_path = os.path.join(home, "grid_imgs") #sys.argv[2]
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    face_predictor_path = os.path.join(os.getcwd(), "utils", "shape_predictor_68_face_landmarks.dat") #sys.argv[3]
    # source_extension = sys.argv[4]
    
    main(source_path, target_path, face_predictor_path)