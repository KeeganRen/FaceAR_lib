#-------------------------------------------------
#
# Project created by QtCreator 2015-03-22T19:52:02
#
#-------------------------------------------------

TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
TEMPLATE = lib
CONFIG += staticlib
CONFIG += release
unix {
    target.path = /usr/lib
    INSTALLS += target
}

QMAKE_CXXFLAGS += \
    -std=c++0x

DESTDIR = ../../libs



#QT       += core

#QT       -= gui

#TARGET = FaceAR_lib
#CONFIG   += console
#CONFIG   -= app_bundle

#TEMPLATE = app

#QMAKE_CXXFLAGS += \
#    -std=c++0x

#DESTDIR = ../../bin


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
    ../src \
    ../src/run/include \
    ../src/FaceAR/include \
    ../src/FaceAR/src

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
    ../src/FaceAR/include/CCNF_patch_expert.h \
    ../src/FaceAR/include/DetectionValidator.h \
    ../src/FaceAR/include/Patch_experts.h \
    ../src/FaceAR/include/PAW.h \
    ../src/FaceAR/include/PDM.h \
    ../src/FaceAR/include/stdafx.h \
    ../src/FaceAR/include/SVR_patch_expert.h \
    ../src/FaceAR/include/FaceAR.h \
    ../src/FaceAR/include/FaceAR_core.h \
    ../src/FaceAR/include/FaceAR_utils.h \
    ../src/FaceAR/include/FaceARParameters.h \
    ../src/FaceAR/include/FaceARTracker.h \
    ../src/dlib/image_processing/frontal_face_detector.h \
    ../src/dlib/opencv.h \
    ../src/dlib/image_processing/frontal_face_detector_abstract.h \
    ../src/dlib/image_processing/object_detector.h \
    ../src/dlib/image_processing/scan_fhog_pyramid.h \
    ../src/dlib/compress_stream.h \
    ../src/dlib/base64.h \
    ../src/dlib/opencv/cv_image.h \
    ../src/dlib/opencv/cv_image_abstract.h \
    ../src/dlib/opencv/to_open_cv.h \
    ../src/dlib/opencv/to_open_cv_abstract.h \
    ../src/run/include/SignalTrackFaceAR.h

SOURCES += \
    ../src/FaceAR/src/CCNF_patch_expert.cpp \
    ../src/FaceAR/src/DetectionValidator.cpp \
    ../src/FaceAR/src/Patch_experts.cpp \
    ../src/FaceAR/src/PAW.cpp \
    ../src/FaceAR/src/PDM.cpp \
    ../src/FaceAR/src/stdafx.cpp \
    ../src/FaceAR/src/SVR_patch_expert.cpp \
    ../src/FaceAR/src/FaceAR.cpp \
    ../src/FaceAR/src/FaceAR_utils.cpp \
    ../src/FaceAR/src/FaceARTracker.cpp \
    ../src/dlib/all/source.cpp \
    ../src/run/src/SignalTrackFaceAR.cpp


