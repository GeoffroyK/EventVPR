import ffmpeg
import argparse

from datetime import datetime

def runConversion(args):
    # Check if a filename was set and create one otherwise
    if args.filename is None:
        args.filename = str(datetime.now().strftime("%m%d-%H-%M"))
    
    (
        ffmpeg
            .input(args.folder_location + '*.png', patter_type='glob', framerate=25)
            .output(args.filename + args.format)
            .run()
    )

    # print(args.folder_location + '*.png')
    # ffmpeg.input(args.folder_location + '*.png', patter_type='glob', framerate=25)
    # ffmpeg.output(args.filename + args.format)
    # ffmpeg.run()

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Images to Video",
        description="Convert images from a specific folder to a video."
    )

    parser.add_argument('--format', type=str, choices=['.mp4', '.avi', '.mjpeg'], default='.mp4',
                        help='File format of the output video')
    parser.add_argument('--folder_location', type=str, required=True,
                        help='Path of the input folder')
    parser.add_argument('--filename', type=str, required=False,
                        help='File name of the video')
    
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parseArgs()
    runConversion(args)
