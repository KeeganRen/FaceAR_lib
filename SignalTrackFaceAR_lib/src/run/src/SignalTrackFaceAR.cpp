
// SimpleFaceAR.cpp : Defines the entry point for the console application.

#include "FaceAR_core.h"

#include <fstream>
#include <sstream>

#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write

using namespace std;
using namespace cv;

int pre_main (std::vector<std::string> &files)
{

//    vector<string> arguments = get_arguments(argc, argv);

    // Some initial parameters that can be overriden from command line
    vector<string> depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files;

    // By default try webcam 0
    int device = 0;

    // cx and cy aren't necessarilly in the image center, so need to be able to override it (start with unit vals and init them if none specified)
    float fx = 500, fy = 500, cx = 0, cy = 0;

    FaceARTracker::FaceARParameters facear_parameters/*(arguments)*/;

    // Get the input output file parameters

    // Indicates that rotation should be with respect to camera plane or with respect to camera
    bool use_camera_plane_pose;
//    FaceARTracker::get_video_input_output_params(files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files, use_camera_plane_pose, arguments);
//    // Get camera parameters
//    FaceARTracker::get_camera_params(device, fx, fy, cx, cy, arguments);

    // The modules that are being used for tracking
    FaceARTracker::FaceAR facear_model(facear_parameters.model_location);

    // If multiple video files are tracked, use this to indicate if we are done
    bool done = false;
    int f_n = -1;

    // If cx (optical axis centre) is undefined will use the image size/2 as an estimate
    bool cx_undefined = false;
    if(cx == 0 || cy == 0)
    {
        cx_undefined = true;
    }

    while(!done) // this is not a for loop as we might also be reading from a webcam
    {

        string current_file;

        // We might specify multiple video files as arguments
        if(files.size() > 0)
        {
            f_n++;
            current_file = files[f_n];
        }
        else
        {
            // If we want to write out from webcam
            f_n = 0;
        }

        bool use_depth = !depth_directories.empty();

        // Do some grabbing
        VideoCapture video_capture;
        if( current_file.size() > 0 )
        {
            std::cout << "Attempting to read from file: " << current_file << std::endl;
            video_capture = VideoCapture( current_file );
        }
        else
        {
            std::cout << "Attempting to capture from device: " << device << std::endl;
            video_capture = VideoCapture( device );

            // Read a first frame often empty in camera
            Mat captured_image;
            video_capture >> captured_image;
        }

        if( !video_capture.isOpened() )
            std::cout << "Fatal error: Failed to open video source" << std::endl;
        else
            std::cout << "Device or file opened" << std::endl;

        Mat captured_image;
        video_capture >> captured_image;

        // If optical centers are not defined just use center of image
        if(cx_undefined)
        {
            cx = captured_image.cols / 2.0f;
            cy = captured_image.rows / 2.0f;
        }

        // Creating output files
        std::ofstream pose_output_file;
        if(!pose_output_files.empty())
        {
            pose_output_file.open (pose_output_files[f_n], ios_base::out);
        }

        std::ofstream landmarks_output_file;
        if(!landmark_output_files.empty())
        {
            landmarks_output_file.open(landmark_output_files[f_n], ios_base::out);
        }

        std::ofstream landmarks_3D_output_file;
        if(!landmark_3D_output_files.empty())
        {
            landmarks_3D_output_file.open(landmark_3D_output_files[f_n], ios_base::out);
        }

        int frame_count = 0;

        // saving the videos
        VideoWriter writerFace;
        if(!tracked_videos_output.empty())
        {
            writerFace = VideoWriter(tracked_videos_output[f_n], CV_FOURCC('D','I','V','X'), 30, captured_image.size(), true);
        }

        // For measuring the timings
        int64 t1,t0 = cv::getTickCount();
        double fps = 10;

        std::cout << "Starting tracking" << std::endl;
        while(!captured_image.empty())
        {

            // Reading the images
            Mat_<float> depth_image;
            Mat_<uchar> grayscale_image;

            if(captured_image.channels() == 3)
            {
                cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
            }
            else
            {
                grayscale_image = captured_image.clone();
            }

            // Get depth image
            if(use_depth)
            {
                char* dst = new char[100];
                std::stringstream sstream;

                sstream << depth_directories[f_n] << "\\depth%05d.png";
                sprintf(dst, sstream.str().c_str(), frame_count + 1);
                // Reading in 16-bit png image representing depth
                Mat_<short> depth_image_16_bit = imread(string(dst), -1);

                // Convert to a floating point depth image
                if(!depth_image_16_bit.empty())
                {
                    depth_image_16_bit.convertTo(depth_image, CV_32F);
                }
                else
                {
                    std::cout << "Warning: " << "Can't find depth image" << std::endl;
                }
            }

            // The actual facial landmark detection / tracking
            bool detection_success = FaceARTracker::DetectLandmarksInVideo(grayscale_image, depth_image, facear_model, facear_parameters);

            // Work out the pose of the head from the tracked model
            Vec6d pose_estimate_FaceAR;
            if(use_camera_plane_pose)
            {
                pose_estimate_FaceAR = FaceARTracker::GetCorrectedPoseCameraPlane(facear_model, fx, fy, cx, cy, facear_parameters);
            }
            else
            {
                pose_estimate_FaceAR = FaceARTracker::GetCorrectedPoseCamera(facear_model, fx, fy, cx, cy, facear_parameters);
            }

            // Visualising the results
            // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
            double detection_certainty = facear_model.detection_certainty;

            double visualisation_boundary = 0.2;

            // Only draw if the reliability is reasonable, the value is slightly ad-hoc
            if(detection_certainty < visualisation_boundary)
            {
                FaceARTracker::Draw(captured_image, facear_model);

                if(detection_certainty > 1)
                    detection_certainty = 1;
                if(detection_certainty < -1)
                    detection_certainty = -1;

                double vis_certainty = (detection_certainty + 1)/(visualisation_boundary +1);

                // A rough heuristic for box around the face width
                int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

                Vec6d pose_estimate_to_draw = FaceARTracker::GetCorrectedPoseCameraPlane(facear_model, fx, fy, cx, cy, facear_parameters);

                // Draw it in reddish if uncertain, blueish if certain
                FaceARTracker::DrawBox(captured_image, pose_estimate_to_draw, Scalar((1-vis_certainty)*255.0,0, vis_certainty*255), thickness, fx, fy, cx, cy);

            }

            // Work out the framerate
            if(frame_count % 10 == 0)
            {
                t1 = cv::getTickCount();
                fps = 10.0 / (double(t1-t0)/cv::getTickFrequency());
                t0 = t1;
            }

            // Write out the framerate on the image before displaying it
            char fpsC[255];
            sprintf(fpsC, "%d", (int)fps);
            string fpsSt("FPS:");
            fpsSt += fpsC;
            cv::putText(captured_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));

            if(!facear_parameters.quiet_mode)
            {
                namedWindow("tracking_result",1);
                imshow("tracking_result", captured_image);

                if(!depth_image.empty())
                {
                    // Division needed for visualisation purposes
                    imshow("depth", depth_image/2000.0);
                }
            }

            // Output the detected facial landmarks
            if(!landmark_output_files.empty())
            {
                landmarks_output_file << frame_count + 1 << " " << detection_success;
                for (int i = 0; i < facear_model.pdm.NumberOfPoints() * 2; ++ i)
                {
                    landmarks_output_file << " " << facear_model.detected_landmarks.at<double>(i);
                }
                landmarks_output_file << endl;
            }

            // Output the detected facial landmarks
            if(!landmark_3D_output_files.empty())
            {
                landmarks_3D_output_file << frame_count + 1 << " " << detection_success;
                Mat_<double> shape_3D = facear_model.GetShape(fx, fy, cx, cy);
                for (int i = 0; i < facear_model.pdm.NumberOfPoints() * 3; ++i)
                {
                    landmarks_3D_output_file << " " << shape_3D.at<double>(i);
                }
                landmarks_3D_output_file << endl;
            }

            // Output the estimated head pose
            if(!pose_output_files.empty())
            {
                double confidence = 0.5 * (1 - detection_certainty);
                pose_output_file << frame_count + 1 << " " << confidence << " " << detection_success << " " << pose_estimate_FaceAR[0] << " " << pose_estimate_FaceAR[1] << " " << pose_estimate_FaceAR[2] << " " << pose_estimate_FaceAR[3] << " " << pose_estimate_FaceAR[4] << " " << pose_estimate_FaceAR[5] << endl;
            }

            // output the tracked video
            if(!tracked_videos_output.empty())
            {
                writerFace << captured_image;
            }

            video_capture >> captured_image;

            // detect key presses
            char character_press = cv::waitKey(1);

            // restart the tracker
            if(character_press == 'r')
            {
                facear_model.Reset();
            }
            // quit the application
            else if(character_press=='q')
            {
                return(0);
            }

            // Update the frame count
            frame_count++;

        }

        frame_count = 0;

        // Reset the model, for the next video
        facear_model.Reset();

        pose_output_file.close();
        landmarks_output_file.close();

        // break out of the loop if done with all the files (or using a webcam)
        if(f_n == files.size() -1 || files.empty())
        {
            done = true;
        }
    }

    return 0;
}
