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

In `/python/test/`, I mainly work on them and copy the final results
to the app folder.

In `/python/dev/` there should be the GUI apps, although they are all
in 1 file so not ideal. Probably good to have them around though.

In `/ModelsYolo/python/app/` there should be the final GUI apps.

- The folder is broken up to `src`, `data` and `conf`. What goes where should be
  self-explanatory.
  - Yolov8l.pt and the theme are modular from the main apps, so a little bit
    more modular
  - Could look into modularise the python applications, though would need to
    wait until the final specs.
- `count_live.py` takes `count_video.py` and modifies it so that it takes a live
  feed from an IP security camera instead of a static file.
- `label.py` helps the user labelling images from a dataset by applying yolov8n.
  The user can then adjust the labelling to their liking and export them to the
  yolo format
- The apps takes themeing from `python/app/conf/theme.qss`.
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
