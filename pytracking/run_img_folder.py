import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


def run_image_folder(tracker_name, tracker_param, img_dir, optional_box=None, debug=None, save_results=False, save_video=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param)
    tracker.run_img_folder(img_dir=img_dir, optional_box=optional_box, debug=debug, save_results=save_results, save_video=save_video)

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('img_folder', type=str, help='path to a image folder.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save inferred video')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_image_folder(args.tracker_name, args.tracker_param,args.img_folder, args.optional_box, args.debug, args.save_results, args.save_video)


if __name__ == '__main__':
    main()