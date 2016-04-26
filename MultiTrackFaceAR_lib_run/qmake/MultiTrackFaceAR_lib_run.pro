#-------------------------------------------------
#
# Project created by QtCreator 2015-03-22T19:52:02
#
#-------------------------------------------------

#TEMPLATE = app
#CONFIG += console
#CONFIG -= app_bundle
#CONFIG -= qt
#TEMPLATE = lib
#CONFIG += staticlib
#CONFIG += release
#unix {
#    target.path = /usr/lib
#    INSTALLS += target
#}

#QMAKE_CXXFLAGS += \
#    -std=c++0x

#DESTDIR = ../../libs



QT       += core

QT       -= gui

TARGET = MultiTrackFaceAR_lib_run
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

QMAKE_CXXFLAGS += \
    -std=c++0x

DESTDIR = ../../bin


INCLUDEPATH += \
    /opt/OpenCV/opencv-3.0/include \
    /opt/OpenCV/opencv-3.0/include/opencv \
    /opt/OpenCV/opencv-3.0/include/opencv2
#    ../src/3rdParty/opencv3/include \
#    ../src/3rdParty/opencv3/include/opencv \
#    ../src/3rdParty/opencv3/include/opencv2

INCLUDEPATH += \
#    /usr/local/include/opencv2 \
#    /usr/local/include/opencv \
#    /usr/local/include \
    /usr/include/boost \
    /usr/include/tbb \
    /usr/include/eigen3


INCLUDEPATH += \
    ../include

LIBS += \
    /home/ren/MyGit/FaceAR_lib/MultiTrackFaceAR_lib_run/libs/libMultiTrackFaceAR_lib.a

LIBS += \
    -ltbb

LIBS += \
    -lboost_filesystem -lboost_system -lboost_thread -lboost_program_options

LIBS += \
    /opt/OpenCV/opencv-3.0/lib/libopencv_calib3d.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_objdetect.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_core.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_videoio.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_features2d.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_viz.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_highgui.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_xfeatures2d.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_imgcodecs.so.3.0 \
    /opt/OpenCV/opencv-3.0/lib/libopencv_imgproc.so.3.0

HEADERS += \
    ../include/MultiTrackFaceAR.h

SOURCES += \
    ../src/FaceAR_main.cpp


