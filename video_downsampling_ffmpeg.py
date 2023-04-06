import os
import subprocess
import numpy as np


def get_filepaths(directory):
        """
        This function will generate the file names in a directory
        tree by walking the tree either top-down or bottom-up. For each
        directory in the tree rooted at directory top (including top itself),
        it yields a 3-tuple (dirpath, dirnames, filenames).
        """
        file_paths = []  # List which will store all of the full filepaths.

        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

        return sorted(file_paths)  # Self-explanatory.


def downsize_video(video_input, video_output):
    # cmds = ['ffmpeg', '-i', video_input, '-s', '568x320', '-c:a', 'copy', video_output]
    # cmds = ['ffmpeg', '-i', video_input, '-s', '960x540', '-c:a', 'copy', video_output]  # (1920x1080) / 2
    cmds = ['ffmpeg', '-i', video_input, '-s', '480x270', '-c:a', 'copy', video_output]  # (1920x1080) / 4
    p = subprocess.Popen(cmds)
    p.wait()


def check_file(file):
    if os.path.exists(file) is True:
        if os.stat(file).st_size > 50000:
            file_valid = True
        else:
            file_valid = False
            os.remove(file)
    else:
        file_valid = False
    return file_valid
#


# downsample videos and keep folder structure

if __name__ == '__main__':
    # input_folder = r"C:\Data\Jenice"
    # output_folder = r"C:\Data\Jenice\downsized"

    # input_folder = r"C:\Data\Ori_BW_MCs"
    # output_folder = r"C:\Data\Ori_BW_MCs\downsized"
    input_folder = r"C:\Users\Gerio\dlc\Touchscreen Chamber-JL-2022-10-26\videos"
    output_folder = r"C:\Users\Gerio\dlc\Touchscreen Chamber-JL-2022-10-26\videos\downsized"

    all_files = get_filepaths(input_folder)
    all_videos = [x for x in all_files if x.lower().endswith('mp4')]

    for video in all_videos:
        save_path = output_folder+video[len(input_folder):]
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        downsize_video(video, save_path)
    #
#
