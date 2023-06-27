import pytube
import os


def cut_video(input_file, output_file, start_time):
    """
    Cuts a video from the specified start and end times.

    Args:
        input_file (str): The path to the input video file.
        output_file (str): The path to the output video file.
        start_time (float): The start time of the cut, in seconds.
        end_time (float): The end time of the cut, in seconds.
    """

    command = f"ffmpeg -i {input_file} -ss {start_time} -vcodec copy -acodec copy {output_file}"

    os.system(command)


video_url_list = [
    "https://www.youtube.com/watch?v=_q0aGyfRl1I&t=40s",
    "https://www.youtube.com/watch?v=Uw746Bv3t_E&t=40s",
    "https://www.youtube.com/watch?v=uSJksKaqT0Q&t=40s",
]
i = 0
for video_url in video_url_list:
    # Create a YouTube object
    i += 1
    youtube = pytube.YouTube(video_url)

    # Get the video stream
    video_stream = youtube.streams.filter(progressive=True, resolution="360p").first()

    # Set the start time to 30 seconds

    # Download the video
    print("Download the video: ", i)

    video_stream.download(f"IO_data/input/video/", filename=f"{i}.mp4")
    print("Cuting video")
    # cut_video(input_file="IO_data/input/video/{i}.mp4", output_file="IO_data/input/video/video_{i}.mp4", start_time=40)
#  ffmpeg -i 3.mp4 -ss 00:00:45 -to 00:53:27 -c:v copy -c:a copy video_3.mp4
