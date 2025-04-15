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

Currently, Yolov8l from Ultralytics is being used, but options such as CSRNet or
RT-DETR are either being explored, this document will mention their use should
they every become the main model in use.

## Graphical demo applications

To make it easy to implement, the models are implemented in simple GUI
applications, which utilises the Qt5 framework to ensure that this application
can be packaged and run n any application, although they are developed and
tested only on Linux thus far.

The applications are themed according to Catppuccin Macchiato using Qt's css
functionalities.

### Overview of the applications

In `/demo_app/test/`, I mainly work on them and copy the final results
to the app folder.

In `/demo_app/dev/` there should be the GUI apps, although they are all
in 1 file so not ideal. Probably good to have them around though.

In `/demo_app/app/` there should be the final GUI apps.

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

### Current state

`coords_video.py` and `count_video.py` works with static videos, although will
break in under any edge case.

### Future plans

- Debug and ensure that the programs work in more edge cases
- Eventually test and compile code into executables for easier deployment
- Test out other viable models (RT-DETR)
- Solve other problems that banks face.
  - To make sure that these programs can do real time tracking, develop the
    ability to take in network webcam instead of static files
- Modularise code for easier debugging and feature adding

## Custom models

Custom models are being fine tuned for specific usecases since in reality, there
are subtle difference between what the raw models provide and what is needed.

### Triggers

The problem with counting people is that some of the time, in fact, a lot of the
time, it would be better if the program doesn't count people, in specific areas
or not. This functionality and indeed any further analysis functionalities would
only needed under a set of conditions. For example, in an ATM, we only care
about the maximum number of people when the ATMs machines are being fed with
cash, or being worked on, while when people are just withdrawing their money, we
can allow unlimited number of people in the premise.

As such, the triggers for these analysis were defined as cash, cash cassettes,
and opened ATMs. Should any of these conditions or a mixture of these conditions,
depending on the premise, occurs, counting people should begin.

To recognise these conditions, from now on would be referred as "triggers", a
separate models and mixture of models is included. The goal is to minimise the
computational costs, or, failing that, at the very least improve the usability
of the models.

### Approach

There are 2 approaches: 1 unified model or 3 separate models, each with their
own pros and cons.

Having 1 unified model simplifies the training process, and since this model
would be running alone, it could theorectically be tuned from a larger model,
potentially improving on accuracy.

The other approach offers more focused and cluster-free models that in theory
more align with the Unix philosophy: Do 1 thing and do it well. It is a more
modular approach that should scale much more easily if there are more triggers
and trigger combinations to come. It comes with a cost of model parameters.
Since whether training for 1 or many classes, the model remains effectively the
same in terms of complexity, the computer systems can only run so much. This
means that the source models have to be smaller, thus potentially compromising
on accuracy.

Currently, YOLO 11 and 8 from Ultralytics are being looked at. Specifically,
m models and lower.

### Current state

There are already 4 models. Yet, due to the limited amount of data, they might
be retrained once new data is available.

They have not also been properly tested.

Testing out on YOLO11. So far the models is not that practical due to inaccurate
results training on sample videos.

Trying out .onnx format. Its crossplatform compatibility is a trade-off against
raw speed.

### Future plans

- Test the existing models
- Retrain them with new data if necessary
- Further apply the models in bank's operations.
