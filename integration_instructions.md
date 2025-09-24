# Neural Blur Integration Guide

## Performance: 1700+ FPS

## Zoom Virtual Camera

# Zoom Virtual Camera Setup:

1. Install v4l2loopback:
   sudo apt-get install v4l2loopback-dkms
   sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="Neural Blur"

2. Run the blur pipeline:
   ffmpeg -f v4l2 -i /dev/video0 -vf "scale=iw/8:ih/8,gblur=sigma=2,scale=iw*8:ih*8" -f v4l2 /dev/video20

3. In Zoom: Settings -> Video -> Camera -> Select "Neural Blur"

## Discord Integration

# Discord Streaming with Blur:

1. Use OBS with neural_blur_obs.py script
2. Set OBS Virtual Camera as output
3. In Discord: Stream -> Change Capture -> OBS Virtual Camera

## Twitch Streaming

# Twitch Streaming with 1700 FPS Blur:

ffmpeg -f v4l2 -i /dev/video0     -vf "scale=iw/8:ih/8,gblur=sigma=2,scale=iw*8:ih*8:flags=fast_bilinear"     -c:v libx264 -preset ultrafast -tune zerolatency -b:v 3000k     -f flv rtmp://live.twitch.tv/live/YOUR_STREAM_KEY

## GStreamer Pipeline

# GStreamer Neural Blur Pipeline
gst-launch-1.0 v4l2src device=/dev/video0 !     video/x-raw,width=1920,height=1080 !     videoscale ! video/x-raw,width=240,height=135 !     videobox blur=5 !     videoscale ! video/x-raw,width=1920,height=1080 !     x264enc speed-preset=ultrafast !     rtmpsink location='rtmp://live.twitch.tv/live/stream_key'
