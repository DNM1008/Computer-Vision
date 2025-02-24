# Utilising Computer Vision models for real life use.

## Summary

This project came from the need to use AI capability to enhance bank security.
There are a number of uses cases for AI Vision, such as decognising suspicious
behaviours, misconducts, unauthorized personel, etc. This project will start out
with the most feaible and arguably most effective use case of all: Counting
people.

There will be other problems being added in the future, in which case this
documents will be updated.

## Models in use

Currently, Yolov8l from ultralytics is being used, but options such as CSRNet or
RT-DETR are either bing explored, this document will mention their use should
they every become the main model in use.

## Graphical application

To make it easy to implement, the models are implemented in simple GUI
applications, which utilises the Qt5 framework to ensure that this application
can be packaged and run n any application, although they are developed and
tested only on Linux thus far.

The applications are themed according to Catppuccin Macchiato using Qt's css
functionalities.

## Overview of the applications

In `/ModelsYolo/python/test/`, I mainly work on them and copy the final results
to the app folder.

- `area_define.py` is used for testing, it should allow the user to import an
  image file and a coordinate file and graph those coordinates on the image,
  highlighting the area covered by the polygon made up by the dots defined by
  the coordinates.
- `coords_image.py` is used to define the coordinates on an image, this could be a
  frame from a video.
- `coords_video.py` is similar to `coords_image.py`, with extra functionalities
  to work with videos as well. This is intended to be 1 of the 2 final products.
- `count_image.py` counts the number of people in an area in an image. The user
  imports the image and coordinate files.
- `count_video.py` and `count_video_rtdert.py` counts the number of people in an
  area in a video. This is the other final products.
- `crop.py` crops the image/video to the smallest frame containing the every set
  of coordinates
- `debug.py` is used for testing, ignore this.

In `/ModelsYolo/python/app/` there should be the final GUI apps.

- `count_live.py` takes `count_video.py` and modifies it so that it takes a live
  feed from an IP security camera instead of a static file.
- Working on a coordinate app as well.

## Current state

`coords_video.py` and `count_video.py` works with static videos, although will
break in under any edge case.

## Future plans

- Debug and ensure that the programs work in more edge cases
- Eventually test and compile code into executables for easier deployment
- Test out other viable models (RT-DETR)
- Solve other problems that banks face.
  - To make sure that these programs can do real time tracking, develop the ability to take in network webcam instead of static files
- Modularise code for easier debugging and feature adding
