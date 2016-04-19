////////////////////////////////////////////////////////////////
/////////
/////////
/////////
/////////
/////////////////////////////////////////////////////////////////

#include "MultiTrackCLM.h"

#include "FaceAR_core.h"

#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <videoio/videoio.hpp>  // Video write
#include <videoio/videoio_c.h>  // Video write

using namespace cv;

void NonOverlapingDetections(const vector<FaceARTracker::FaceAR>& facear_models, vector<Rect_<double> >& face_detections)
{

    // Go over the model and eliminate detections that are not informative (there already is a tracker there)
    for(size_t model = 0; model < facear_models.size(); ++model)
    {

        // See if the detections intersect
        Rect_<double> model_rect = facear_models[model].GetBoundingBox();

        for(int detection = face_detections.size()-1; detection >=0; --detection)
        {
            double intersection_area = (model_rect & face_detections[detection]).area();
            double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

            // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
            if( intersection_area/union_area > 0.5)
            {
                face_detections.erase(face_detections.begin() + detection);
            }
        }
    }
}

int pre_main (std::vector<std::string> &files)
{

    ///vector<string> arguments; //= get_arguments(argc, argv);

    // Some initial parameters that can be overriden from command line
    vector<string> depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files;

    // By default try webcam 0
    int device = 0;

    // cx and cy aren't necessarilly in the image center, so need to be able to override it (start with unit vals and init them if none specified)
    float fx = 600, fy = 600, cx = 0, cy = 0;

    FaceARTracker::FaceARParameters facear_params/*(arguments)*/;
    facear_params.use_face_template = true;
    facear_params.track_gaze = true;
    // This is so that the model would not try re-initialising itself
    facear_params.reinit_video_every = -1;

    facear_params.curr_face_detector = FaceARTracker::FaceARParameters::HOG_SVM_DETECTOR;

    vector<FaceARTracker::FaceARParameters> facear_parameters;
    facear_parameters.push_back(facear_params);

    // Get the input output file parameters
    ///bool use_camera_plane_pose;
    ///FaceARTracker::get_video_input_output_params(files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files, use_camera_plane_pose, arguments);
    // Get camera parameters
    ///FaceARTracker::get_camera_params(device, fx, fy, cx, cy, arguments);

    // The modules that are being used for tracking
    vector<FaceARTracker::FaceAR> facear_models;
    vector<bool> active_models;

    int num_faces_max = 4;

    FaceARTracker::FaceAR facear_model(facear_parameters[0].model_location);
    facear_model.face_detector_HAAR.load(facear_parameters[0].face_detector_location);
    facear_model.face_detector_location = facear_parameters[0].face_detector_location;

    facear_models.reserve(num_faces_max);

    facear_models.push_back(facear_model);
    active_models.push_back(false);

    for (int i = 1; i < num_faces_max; ++i)
    {
        facear_models.push_back(facear_model);
        active_models.push_back(false);
        facear_parameters.push_back(facear_params);
    }

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
        {
            std::cout << "Failed to open video source" << std::endl;
            abort();
        }
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
            pose_output_file.open (pose_output_files[f_n]);
        }

        std::ofstream landmarks_output_file;
        if(!landmark_output_files.empty())
        {
            landmarks_output_file.open(landmark_output_files[f_n]);
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

            Mat disp_image = captured_image.clone();

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

            vector<Rect_<double> > face_detections;

            bool all_models_active = true;
            for(unsigned int model = 0; model < facear_models.size(); ++model)
            {
                if(!active_models[model])
                {
                    all_models_active = false;
                }
            }

            // Get the detections (every 8th frame and when there are free models available for tracking)
            if(frame_count % 8 == 0 && !all_models_active)  //8 Keegan.Ren
            {
                if(facear_parameters[0].curr_face_detector == FaceARTracker::FaceARParameters::HOG_SVM_DETECTOR)
                {
                    vector<double> confidences;
                    FaceARTracker::DetectFacesHOG(face_detections, grayscale_image, facear_models[0].face_detector_HOG, confidences);
                }
                else
                {
                    FaceARTracker::DetectFaces(face_detections, grayscale_image, facear_models[0].face_detector_HAAR);
                }

            }

            // Keep only non overlapping detections (also convert to a concurrent vector
            NonOverlapingDetections(facear_models, face_detections);

            vector<tbb::atomic<bool> > face_detections_used(face_detections.size());

            // Go through every model and update the tracking TODO pull out as a separate parallel/non-parallel method
            tbb::parallel_for(0, (int)facear_models.size(), [&](int model){
            //for(unsigned int model = 0; model < facear_models.size(); ++model)
            //{

                bool detection_success = false;

                // If the current model has failed more than 4 times in a row, remove it
                if(facear_models[model].failures_in_a_row > 4)
                {
                    active_models[model] = false;
                    facear_models[model].Reset();

                }

                // If the model is inactive reactivate it with new detections
                if(!active_models[model])
                {

                    for(size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
                    {
                        // if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
                        if(face_detections_used[detection_ind].compare_and_swap(true, false) == false)
                        {

                            // Reinitialise the model
                            facear_models[model].Reset();

                            // This ensures that a wider window is used for the initial landmark localisation
                            facear_models[model].detection_success = false;
                            detection_success = FaceARTracker::DetectLandmarksInVideo(grayscale_image, depth_image, face_detections[detection_ind], facear_models[model], facear_parameters[model]);

                            // This activates the model
                            active_models[model] = true;

                            // break out of the loop as the tracker has been reinitialised
                            break;
                        }

                    }
                }
                else
                {
                    // The actual facial landmark detection / tracking
                    detection_success = FaceARTracker::DetectLandmarksInVideo(grayscale_image, depth_image, facear_models[model], facear_parameters[model]);
                }
            });

            // Go through every model and visualise the results
            for(size_t model = 0; model < facear_models.size(); ++model)
            {
                // Visualising the results
                // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
                double detection_certainty = facear_models[model].detection_certainty;

                double visualisation_boundary = -0.1;

                // Only draw if the reliability is reasonable, the value is slightly ad-hoc
                if(detection_certainty < visualisation_boundary)
                {
                    FaceARTracker::Draw(disp_image, facear_models[model]);

                    if(detection_certainty > 1)
                        detection_certainty = 1;
                    if(detection_certainty < -1)
                        detection_certainty = -1;

                    detection_certainty = (detection_certainty + 1)/(visualisation_boundary +1);

                    // A rough heuristic for box around the face width
                    int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

                    // Work out the pose of the head from the tracked model
                    Vec6d pose_estimate_FaceAR = FaceARTracker::GetCorrectedPoseCameraPlane(facear_models[model], fx, fy, cx, cy, facear_parameters[model]);

                    // Draw it in reddish if uncertain, blueish if certain
                    FaceARTracker::DrawBox(disp_image, pose_estimate_FaceAR, Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);
                }
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
            cv::putText(disp_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));

            int num_active_models = 0;

            for( size_t active_model = 0; active_model < active_models.size(); active_model++)
            {
                if(active_models[active_model])
                {
                    num_active_models++;
                }
            }

            char active_m_C[255];
            sprintf(active_m_C, "%d", num_active_models);
            string active_models_st("Active models:");
            active_models_st += active_m_C;
            cv::putText(disp_image, active_models_st, cv::Point(10,60), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));

            if(!facear_parameters[0].quiet_mode)
            {
                namedWindow("tracking_result",1);
                imshow("tracking_result", disp_image);

                if(!depth_image.empty())
                {
                    // Division needed for visualisation purposes
                    imshow("depth", depth_image/2000.0);
                }
            }

            // output the tracked video
            if(!tracked_videos_output.empty())
            {
                writerFace << disp_image;
            }

            video_capture >> captured_image;

            // detect key presses
            char character_press = cv::waitKey(1);

            // restart the trackers
            if(character_press == 'r')
            {
                for(size_t i=0; i < facear_models.size(); ++i)
                {
                    facear_models[i].Reset();
                    active_models[i] = false;
                }
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
        for(size_t model=0; model < facear_models.size(); ++model)
        {
            facear_models[model].Reset();
            active_models[model] = false;
        }
        pose_output_file.close();
        landmarks_output_file.close();

        // break out of the loop if done with all the files
        if(f_n == files.size() -1)
        {
            done = true;
        }
    }

    return 0;
}


