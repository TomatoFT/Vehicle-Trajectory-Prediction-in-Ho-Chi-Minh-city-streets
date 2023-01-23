from moviepy.video.io.VideoFileClip import VideoFileClip

# Open the video file
clip = VideoFileClip("IO_data/input/video/city.mp4")

# Define the start and end times for the first and second clips
t1 = 0 # start time of the first clip
t2 = 420 # end time of the first clip
t3 = 502 # start time of the second clip
t4 = clip.duration # end time of the second clip

# Extract the clips
clip1 = clip.subclip(t2, t3)
# clip2 = clip.subclip(t3, t4)

# Write the clips to new files
clip1.write_videofile("IO_data/output/city_1.mp4")
# clip2.write_videofile("IO_data/output/city_2.mp4")
