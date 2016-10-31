/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: examples/pose_estimation.cpp
 *
 * Copyright 2014, 2015 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ImageProcess.h"
/* face landmark */
#include "frontalUtil.h"
#include <io.h>

#include "superviseddescent/superviseddescent.hpp"
#include "superviseddescent/regressors.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif

#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <string.h>
#include <cstdio>
using namespace superviseddescent;
using cv::Mat;
using cv::Vec4f;
using std::vector;
using std::cout;
using std::endl;

/* face landmark */
typedef struct _finddata_t  FILE_SEARCH;

float mod(float a, float N) { return a - N*floor(a / N); }

void originFromYML(const std::string& fileName, Mat_<float>& facialFeaturePoint)
{
	FileStorage fs(fileName, FileStorage::READ);

	if (!fs.isOpened()){
		string msg = "open " + fileName + " error";
		DEBUGMSG(msg);
		return;
	}

	fs["facialFeaturePoint"] >> facialFeaturePoint;
	fs.release();
}

void readFacialFeaturePointFromYML(const std::string& fileName, Mat_<float>& facialFeaturePoint, float* standardX, float* standardY, float* NormalizeIndex)
{
	/* normalize parameters */
	int eyeStartIndex1 = 39, eyeStartIndex2 = 42;
	int eyeEndIndex = 45;
	//standardX 눈 사이 중앙에서 오른쪽 눈 끝쪽까지의 거리를 100으로 두고
	
	FileStorage fs(fileName,FileStorage::READ);

	if(!fs.isOpened()){
		string msg = "open "  + fileName + " error";
		DEBUGMSG(msg);
		return;
	}

	fs["facialFeaturePoint"] >> facialFeaturePoint;
	
	//facialFeaturePoints normalize
	*standardX = (facialFeaturePoint[eyeStartIndex1][0] + facialFeaturePoint[eyeStartIndex2][0]) / 2;
	*standardY = (facialFeaturePoint[eyeStartIndex1][1] + facialFeaturePoint[eyeStartIndex2][1]) / 2;
	*NormalizeIndex = 100 / (facialFeaturePoint[eyeEndIndex][0] - *standardX);

	for (int i = 0; i < 68; i++){
		facialFeaturePoint[i][0] = (facialFeaturePoint[i][0] - *standardX) * *NormalizeIndex;
		facialFeaturePoint[i][1] = (facialFeaturePoint[i][1] - *standardY) * *NormalizeIndex;
	}
	fs.release();
}
/* face landmark */
#pragma region

float rad2deg(float radians)
{
	return radians * static_cast<float>(180 / CV_PI);
}

float deg2rad(float degrees)
{
	return degrees * static_cast<float>(CV_PI / 180);
}

float focalLengthToFovy(float focalLength, float height)
{
	return rad2deg(2.0f * std::atan2(height, 2.0f * focalLength));
}


/**
 * Creates a 4x4 rotation matrix with a rotation in \p angle radians
 * around the x axis (pitch angle). Following OpenGL and Qt's conventions.
 *
 * @param[in] angle The angle around the x axis in radians.
 */
cv::Mat createRotationMatrixX(float angle)
{
	cv::Mat rotX = (cv::Mat_<float>(4, 4) <<
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, std::cos(angle), -std::sin(angle), 0.0f,
		0.0f, std::sin(angle), std::cos(angle), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return rotX;
}

/**
 * Creates a 4x4 rotation matrix with a rotation in \p angle radians
 * around the y axis (yaw angle). Following OpenGL and Qt's conventions.
 *
 * @param[in] angle The angle around the y axis in radians.
 */
cv::Mat createRotationMatrixY(float angle)
{
	cv::Mat rotY = (cv::Mat_<float>(4, 4) <<
		std::cos(angle), 0.0f, std::sin(angle), 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		-std::sin(angle), 0.0f, std::cos(angle), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return rotY;
}

/**
 * Creates a 4x4 rotation matrix with a rotation in \p angle radians
 * around the z axis (roll angle). Following OpenGL and Qt's conventions.
 *
 * @param[in] angle The angle around the z axis in radians.
 */
cv::Mat createRotationMatrixZ(float angle)
{
	cv::Mat rotZ = (cv::Mat_<float>(4, 4) <<
		std::cos(angle), -std::sin(angle), 0.0f, 0.0f,
		std::sin(angle), std::cos(angle), 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return rotZ;
}

/**
 * Creates a 4x4 scaling matrix that scales homogeneous vectors along x, y and z.
 *
 * @param[in] sx Scaling along the x direction.
 * @param[in] sy Scaling along the y direction.
 * @param[in] sz Scaling along the z direction.
 */
cv::Mat createScalingMatrix(float sx, float sy, float sz)
{
	cv::Mat scaling = (cv::Mat_<float>(4, 4) <<
		sx, 0.0f, 0.0f, 0.0f,
		0.0f, sy, 0.0f, 0.0f,
		0.0f, 0.0f, sz, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	return scaling;
}

/**
 * Creates a 4x4 translation matrix that scales homogeneous vectors along x, y and z direction.
 *
 * @param[in] tx Translation along the x direction.
 * @param[in] ty Translation along the y direction.
 * @param[in] tz Translation along the z direction.
 */
cv::Mat createTranslationMatrix(float tx, float ty, float tz)
{
	cv::Mat translation = (cv::Mat_<float>(4, 4) <<
		1.0f, 0.0f, 0.0f, tx,
		0.0f, 1.0f, 0.0f, ty,
		0.0f, 0.0f, 1.0f, tz,
		0.0f, 0.0f, 0.0f, 1.0f);
	return translation;
}

/**
 * Creates a 4x4 perspective projection matrix, following OpenGL & Qt conventions.
 *
 * @param[in] verticalAngle Vertical angle of the FOV, in degrees.
 * @param[in] aspectRatio Aspect ratio of the camera.
 * @param[in] n Near plane distance.
 * @param[in] f Far plane distance.
 */
cv::Mat createPerspectiveProjectionMatrix(float verticalAngle, float aspectRatio, float n, float f)
{
	float radians = (verticalAngle / 2.0f) * static_cast<float>(CV_PI) / 180.0f;
	float sine = std::sin(radians);
	// if sinr == 0.0f, return, something wrong
	float cotan = std::cos(radians) / sine;
	cv::Mat perspective = (cv::Mat_<float>(4, 4) <<
		cotan / aspectRatio, 0.0f, 0.0f, 0.0f,
		0.0f, cotan, 0.0f, 0.0f,
		0.0f, 0.0f, -(n + f) / (f - n), (-2.0f * n * f) / (f - n),
		0.0f, 0.0f, -1.0f, 0.0f);
	return perspective;
}

/**
 * Projects a vertex in homogeneous coordinates to screen coordinates.
 *
 * @param[in] vertex A vertex in homogeneous coordinates.
 * @param[in] model_view_projection A 4x4 model-view-projection matrix.
 * @param[in] screen_width Screen / window width.
 * @param[in] screen_height Screen / window height.
 */
cv::Vec3f projectVertex(cv::Vec4f vertex, cv::Mat model_view_projection, int screen_width, int screen_height)
{
	cv::Mat clipSpace = model_view_projection * cv::Mat(vertex);
	cv::Vec4f clipSpaceV(clipSpace);
	clipSpaceV = clipSpaceV / clipSpaceV[3]; // divide by w
	// Viewport transform:
	float x_ss = (clipSpaceV[0] + 1.0f) * (screen_width / 2.0f);
	float y_ss = screen_height - (clipSpaceV[1] + 1.0f) * (screen_height / 2.0f); // flip the y axis, image origins are different
	cv::Vec2f screenSpace(x_ss, y_ss);
	return cv::Vec3f(screenSpace[0], screenSpace[1], clipSpaceV[2]);
}

/**
 * Function object that projects a 3D model to 2D, given a set of
 * parameters [R_x, R_y, R_z, t_x, t_y, t_z]. 
 *
 * It is used in this example for the 6 DOF pose estimation.
 * A perspective camera with focal length 1800 and a screen of
 * 1000 x 1000, with the camera at [500, 500], is used in this example.
 *
 * The 2D points are normalised after projection by subtracting the camera
 * origin and dividing by the focal length.
 */
class ModelProjection
{
public:
	/**
	 * Constructs a new projection object with the given model.
	 *
	 * @param[in] model 3D model points in a 4 x n matrix, where n is the number of points, and the points are in homgeneous coordinates.
	 */
	ModelProjection(cv::Mat model) : model(model)
	{
	};

	/**
	 * Uses the current parameters ([R, t], in SDM terminology the \c x) to
	 * project the points from the 3D model to 2D space. The 2D points are
	 * the new \c y.
	 *
	 * The current parameter estimate is given as a row vector
	 * [R_x, R_y, R_z, t_x, t_y, t_z].
	 *
	 * @param[in] parameters The current estimate of the 6 DOF.
	 * @param[in] regressorLevel Not used in this example.
	 * @param[in] trainingIndex Not used in this example.
	 * @return Returns normalised projected 2D landmark coordinates.
	 */
	cv::Mat operator()(cv::Mat parameters, size_t regressorLevel, int trainingIndex = 0)
	{
		assert((parameters.cols == 1 && parameters.rows == 6) || (parameters.cols == 6 && parameters.rows == 1));
		using cv::Mat;
		// Project the 3D model points using the current parameters:
		float focalLength = 1800.0f;
		Mat rotPitchX = createRotationMatrixX(deg2rad(parameters.at<float>(0)));
		Mat rotYawY = createRotationMatrixY(deg2rad(parameters.at<float>(1)));
		Mat rotRollZ = createRotationMatrixZ(deg2rad(parameters.at<float>(2)));
		Mat translation = createTranslationMatrix(parameters.at<float>(3), parameters.at<float>(4), parameters.at<float>(5));
		Mat modelMatrix = translation * rotYawY * rotPitchX * rotRollZ;
		const float aspect = static_cast<float>(1000) / static_cast<float>(1000);
		float fovY = focalLengthToFovy(focalLength, 1000);
		Mat projectionMatrix = createPerspectiveProjectionMatrix(fovY, aspect, 1.0f, 5000.0f);

		int numLandmarks = model.cols;
		Mat new2dProjections(1, numLandmarks * 2, CV_32FC1);
		for (int lm = 0; lm < numLandmarks; ++lm) {
			cv::Vec3f vtx2d = projectVertex(cv::Vec4f(model.col(lm)), projectionMatrix * modelMatrix, 1000, 1000);
			// 'Normalise' the image coordinates of the projection by subtracting the origin (center of the image) and dividing by f:
			vtx2d = (vtx2d - cv::Vec3f(500.0f, 500.0f)) / focalLength;
			new2dProjections.at<float>(lm) = vtx2d[0]; // the x coord
			new2dProjections.at<float>(lm + numLandmarks) = vtx2d[1]; // y coord
		}
		return new2dProjections;
	};
private:
	cv::Mat model;
};

#pragma endregion function for face pose estimantion

/**
 * This app demonstrates learning of the descent direction from data for
 * a simple 6 degree of freedom face pose estimation.
 *
 * It uses a simple 10-point 3D face model, generates random poses and
 * uses the generated pose parameters (\c x) and their respective 2D
 * projections (\c y) to learn how to optimise for the parameters given
 * input landmarks \c y.
 *
 * This is an example of the library when a known template \c y is available
 * for training and testing.
 */

int main(int argc, char *argv[])
{
	// image frontalization 
	Mat image = imread("capture.jpg");
	Mat frontalImage;

	// face landmark 
	Mat_<float> facialFeaturePoints;
	Mat_<float> inputFeaturePoints;
	Mat_<float> standardFeaturePoints;
	Mat_<float> originFeaturePoints;

	intptr_t h_file;
	FILE_SEARCH file_search;
	int errorhandler = 0;
	char path[100] = "./KKU";
	char path2[100];
	char readfile[100];

	char inputfile[100] = "capture_original.yml";
	float normalization;
	float pre_diff = 100000000000;
	float diff;
	char temp[100];

	// restoration
	float inp_x = 0, inp_y = 0, i_idx = 0;
	float std_x = 0, std_y = 0, n_idx = 0;
	float tmp_x = 0, tmp_y = 0, t_idx = 0;

	Mat facemodel; // The point numbers are from the iBug landmarking scheme
	facemodel.push_back(Vec4f(-0.287526f, -2.0203f, 3.33725f, 1.0f)); // nose tip, 31
	facemodel.push_back(Vec4f(-0.11479f, -17.2056f, -13.5569f, 1.0f)); // nose-lip junction, 34
	facemodel.push_back(Vec4f(-46.1668f, 34.7219f, -35.938f, 1.0f)); // right eye outer corner, 37
	facemodel.push_back(Vec4f(-18.926f, 31.5432f, -29.9641f, 1.0f)); // right eye inner corner, 40
	facemodel.push_back(Vec4f(19.2574f, 31.5767f, -30.229f, 1.0f)); // left eye inner corner, 43
	facemodel.push_back(Vec4f(46.1914f, 34.452f, -36.1317f, 1.0f)); // left eye outer corner, 46
	facemodel.push_back(Vec4f(-23.7552f, -35.7461f, -28.2573f, 1.0f)); // mouth right corner, 49
	facemodel.push_back(Vec4f(-0.0753515f, -28.3064f, -12.8984f, 1.0f)); // upper lip center top, 52
	facemodel.push_back(Vec4f(23.7138f, -35.7886f, -28.5949f, 1.0f)); // mouth left corner, 55
	facemodel.push_back(Vec4f(0.125511f, -44.7427f, -17.1411f, 1.0f)); // lower lip center bottom, 58
	facemodel = facemodel.reshape(1, 10).t(); // reshape to 1 channel, 10 rows, then transpose

	// Random generator for a random angle in [-30, 30]:
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> angle_distribution(-30, 30);
	auto random_angle = [&angle_distribution, &engine]() { return angle_distribution(engine); };

	// For the sake of a brief example, we only sample the angles and keep the x and y translation constant:
	float tx = 0.0f;
	float ty = 0.0f;
	float tz = -2000.0f;

	vector<LinearRegressor<>> regressors;
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));
	regressors.emplace_back(LinearRegressor<>(Regulariser(Regulariser::RegularisationType::MatrixNorm, 2.0f, true)));

	SupervisedDescentOptimiser<LinearRegressor<>> supervised_descent_model(regressors);

	ModelProjection projection(facemodel);

	// Generate 500 random parameter samples, consisting of
	// the 6 DOF. Stored as [r_x, r_y, r_z, t_x, t_y, t_z]:
	int num_samples = 500;
	Mat x_tr(num_samples, 6, CV_32FC1);
	for (int row = 0; row < num_samples; ++row) {
		x_tr.at<float>(row, 0) = random_angle();
		x_tr.at<float>(row, 1) = random_angle();
		x_tr.at<float>(row, 2) = random_angle();
		x_tr.at<float>(row, 3) = tx;
		x_tr.at<float>(row, 4) = ty;
		x_tr.at<float>(row, 5) = tz;
	}

	// Calculate and store the corresponding 2D landmark projections:
	// Note: In a real application, we would add some noise to the data.
	auto num_landmarks = facemodel.cols;
	Mat y_tr(num_samples, 2 * num_landmarks, CV_32FC1);
	for (int row = 0; row < num_samples; ++row) {
		auto landmarks = projection(x_tr.row(row), 0);
		landmarks.copyTo(y_tr.row(row));
	}

	Mat x0 = Mat::zeros(num_samples, 6, CV_32FC1); // fixed initialisation of the parameters, all zero, except t_z
	x0.col(5) = -2000.0f;

	auto print_residual = [&x_tr](const cv::Mat& current_predictions) {
		cout << cv::norm(current_predictions, x_tr, cv::NORM_L2) / cv::norm(x_tr, cv::NORM_L2) << endl;
	};
	// Train the model. We'll also specify an optional callback function:
	cout << "Training the model, printing the residual after each learned regressor: " << endl;
	supervised_descent_model.train(x_tr, x0, y_tr, projection, print_residual);

	// Test: Omitted for brevity, as we didn't generate test data
	//Mat predictions = supervisedDescentModel.test(x0, y_tr, projection, printResidual);

	// receive input
	readFacialFeaturePointFromYML(inputfile, inputFeaturePoints, &inp_x, &inp_y, &i_idx);

	// Prediction on new landmarks, [x_0, ..., x_n, y_0, ..., y_n]:
	//Mat landmarks = (cv::Mat_<float>(1, 20) << 498.0f, 504.0f, 479.0f, 498.0f, 529.0f, 553.0f, 489.0f, 503.0f, 527.0f, 503.0f, 502.0f, 513.0f, 457.0f, 465.0f, 471.0f, 471.0f, 522.0f, 522.0f, 530.0f, 536.0f);
	Mat landmarks = (cv::Mat_<float>(1, 20) << inputFeaturePoints[30][0], inputFeaturePoints[33][0], inputFeaturePoints[36][0], inputFeaturePoints[39][0], inputFeaturePoints[42][0],
		inputFeaturePoints[45][0], inputFeaturePoints[48][0], inputFeaturePoints[51][0], inputFeaturePoints[54][0], inputFeaturePoints[57][0],
		inputFeaturePoints[30][0], inputFeaturePoints[33][1], inputFeaturePoints[36][1], inputFeaturePoints[39][1], inputFeaturePoints[42][1],
		inputFeaturePoints[45][0], inputFeaturePoints[48][1], inputFeaturePoints[51][1], inputFeaturePoints[54][1], inputFeaturePoints[57][1]);
	// Normalise the coordinates w.r.t. the image origin and focal length (we do the same during training):
	normalization = inputFeaturePoints[30][0] + inputFeaturePoints[33][0] + inputFeaturePoints[36][0] + inputFeaturePoints[39][0] + inputFeaturePoints[42][0] +
		inputFeaturePoints[45][0] + inputFeaturePoints[48][0] + inputFeaturePoints[51][0] + inputFeaturePoints[54][0] + inputFeaturePoints[57][0] +
		inputFeaturePoints[30][0] + inputFeaturePoints[33][1] + inputFeaturePoints[36][1] + inputFeaturePoints[39][1] + inputFeaturePoints[42][1] +
		inputFeaturePoints[45][0] + inputFeaturePoints[48][1] + inputFeaturePoints[51][1] + inputFeaturePoints[54][1] + inputFeaturePoints[57][1];
	normalization /= 20;

	landmarks = (landmarks - normalization) / 1800.0f;

	Mat initial_params = Mat::zeros(1, 6, CV_32FC1);
	initial_params.at<float>(5) = -2000.0f; // [0, 0, 0, 0, 0, -2000]

	Mat input_params = supervised_descent_model.predict(initial_params, landmarks, projection);
	//	cout << "Groundtruth pose: pitch = 11.0, yaw = -25.0, roll = -10.0" << endl;
	input_params.at<float>(0) = mod(input_params.at<float>(0), 2000);
	input_params.at<float>(1) = mod(input_params.at<float>(1), 2000);

	cout << "input Predicted pose: pitch = " << input_params.at<float>(0) << ", yaw = " << input_params.at<float>(1) << endl; //", roll = " << input_params.at<float>(2) << endl;

	// repeat
	for (int j = 0; j < 145; j++){
		sprintf(path2, "%s/%03d", path, j);
		sprintf(readfile, "%s/*.yml", path2);

		if ((h_file = _findfirst(readfile, &file_search)) == -1L) {
			printf("No files in current directory!\n");

		}
		else {
			do {
				//file read
				if (errorhandler > 1){

					sprintf(readfile, "%s/%s", path2, file_search.name);
					readFacialFeaturePointFromYML(readfile, facialFeaturePoints, &tmp_x, &tmp_y, &t_idx);

					// Prediction on new landmarks, [x_0, ..., x_n, y_0, ..., y_n]:
					//Mat landmarks = (cv::Mat_<float>(1, 20) << 498.0f, 504.0f, 479.0f, 498.0f, 529.0f, 553.0f, 489.0f, 503.0f, 527.0f, 503.0f, 502.0f, 513.0f, 457.0f, 465.0f, 471.0f, 471.0f, 522.0f, 522.0f, 530.0f, 536.0f);
					Mat landmarks = (cv::Mat_<float>(1, 20) << facialFeaturePoints[30][0], facialFeaturePoints[33][0], facialFeaturePoints[36][0], facialFeaturePoints[39][0], facialFeaturePoints[42][0],
						facialFeaturePoints[45][0], facialFeaturePoints[48][0], facialFeaturePoints[51][0], facialFeaturePoints[54][0], facialFeaturePoints[57][0],
						facialFeaturePoints[30][0], facialFeaturePoints[33][1], facialFeaturePoints[36][1], facialFeaturePoints[39][1], facialFeaturePoints[42][1],
						facialFeaturePoints[45][0], facialFeaturePoints[48][1], facialFeaturePoints[51][1], facialFeaturePoints[54][1], facialFeaturePoints[57][1]);
					// Normalise the coordinates w.r.t. the image origin and focal length (we do the same during training):
					normalization = facialFeaturePoints[30][0] + facialFeaturePoints[33][0] + facialFeaturePoints[36][0] + facialFeaturePoints[39][0] + facialFeaturePoints[42][0] +
						facialFeaturePoints[45][0] + facialFeaturePoints[48][0] + facialFeaturePoints[51][0] + facialFeaturePoints[54][0] + facialFeaturePoints[57][0] +
						facialFeaturePoints[30][0] + facialFeaturePoints[33][1] + facialFeaturePoints[36][1] + facialFeaturePoints[39][1] + facialFeaturePoints[42][1] +
						facialFeaturePoints[45][0] + facialFeaturePoints[48][1] + facialFeaturePoints[51][1] + facialFeaturePoints[54][1] + facialFeaturePoints[57][1];
					normalization /= 20;

					landmarks = (landmarks - normalization) / 1800.0f;

					Mat initial_params = Mat::zeros(1, 6, CV_32FC1);
					initial_params.at<float>(5) = -2000.0f; // [0, 0, 0, 0, 0, -2000]

					Mat predicted_params = supervised_descent_model.predict(initial_params, landmarks, projection);

					//	cout << "Groundtruth pose: pitch = 11.0, yaw = -25.0, roll = -10.0" << endl;
					//  cout << "Predicted pose: pitch = " << predicted_params.at<float>(0) << ", yaw = " << predicted_params.at<float>(1) << ", roll = " << predicted_params.at<float>(2) << endl;



					//**** YAW는 되는데 PITCH 안됨
					//start normalization when meet the condition
					if (abs(predicted_params.at<float>(0) - input_params.at<float>(0)) < 100 || abs(predicted_params.at<float>(1) - input_params.at<float>(1)) < 10){
						//cout << predicted_params.at<float>(0) - input_params.at<float>(0) << " " << predicted_params.at<float>(1) - input_params.at<float>(1) << endl;

						diff = 0;
						// least mean square 
						for (int i = 0; i < 17; i++){
							diff += (inputFeaturePoints[i][0] - facialFeaturePoints[i][0]) * (inputFeaturePoints[i][0] - facialFeaturePoints[i][0]) + (inputFeaturePoints[i][1] - facialFeaturePoints[i][1]) * (inputFeaturePoints[i][1] - facialFeaturePoints[i][1]);
						}
						if (diff < pre_diff){
							strcpy(temp, readfile);
							pre_diff = diff;
						}
					}
				}
				errorhandler++;

			} while (_findnext(h_file, &file_search) == 0);

			_findclose(h_file);
		}
	}
	cout << temp << endl;


	//Part 2
	readFacialFeaturePointFromYML(temp, facialFeaturePoints, &tmp_x, &tmp_y, &t_idx); //가장 비슷했던 애
	strtok(temp, "N");
	sprintf(temp, "%sstandard.yml", temp);
	readFacialFeaturePointFromYML(temp, standardFeaturePoints, &std_x, &std_y, &n_idx); //그 애의 standard 좌표

	//inputFeaturePoints 최종 움직여야할 위치
	for (int i = 0; i < 68; i++){									//standard					most similar one
		inputFeaturePoints[i][0] = (inputFeaturePoints[i][0] + standardFeaturePoints[i][0] - facialFeaturePoints[i][0]) / i_idx + inp_x;
		inputFeaturePoints[i][1] = (inputFeaturePoints[i][1] + standardFeaturePoints[i][1] - facialFeaturePoints[i][1]) / i_idx + inp_y;
	}

	originFromYML(inputfile, originFeaturePoints); //원본 inputFeaturePoint

	int byte_size = image.total() * image.elemSize();
	BYTE * bytes = new BYTE[byte_size];  // you will have to delete[] that later
	BYTE * newbytes = new BYTE[byte_size];
	std::memcpy(bytes, image.data, byte_size * sizeof(BYTE));
	int sourceX[81], sourceY[81], destX[81], destY[81];

	for (int i = 0; i < 81; i++)
	{
		sourceX[i] = -1;
		sourceY[i] = -1;
		destX[i] = -1;
		destY[i] = -1;
	}

	for (int i = 0; i < 9; i++){
		sourceY[i] = 0;
		destY[i] = 0;
	}
	for (int i = 72; i < 81; i++){
		sourceY[i] = image.size().height;
		destY[i] = image.size().height;
	}

	for (int i = 0; i < 81; i += 9){
		sourceX[i] = 0;
		destX[i] = 0;
	}
	for (int i = 8; i < 81; i += 9){
		sourceX[i] = image.size().width;
		destX[i] = image.size().width;
	}

#pragma region


	sourceX[1] = originFeaturePoints[0][0];
	sourceX[2] = originFeaturePoints[17][0];
	sourceX[3] = originFeaturePoints[20][0];
	sourceX[4] = (originFeaturePoints[21][0] + originFeaturePoints[22][0]) / 2;
	sourceX[5] = originFeaturePoints[23][0];
	sourceX[6] = originFeaturePoints[26][0];
	sourceX[7] = originFeaturePoints[16][0];

	sourceX[73] = originFeaturePoints[4][0];
	sourceX[74] = originFeaturePoints[5][0];
	sourceX[75] = originFeaturePoints[6][0];
	sourceX[76] = originFeaturePoints[8][0];
	sourceX[77] = originFeaturePoints[10][0];
	sourceX[78] = originFeaturePoints[11][0];
	sourceX[79] = originFeaturePoints[12][0];

	sourceY[9] = originFeaturePoints[17][1];
	sourceY[18] = originFeaturePoints[0][1];
	sourceY[27] = originFeaturePoints[1][1];
	sourceY[36] = originFeaturePoints[2][1];
	sourceY[45] = originFeaturePoints[3][1];
	sourceY[54] = originFeaturePoints[4][1];
	sourceY[63] = originFeaturePoints[5][1];

	sourceY[17] = originFeaturePoints[26][1];
	sourceY[26] = originFeaturePoints[16][1];
	sourceY[35] = originFeaturePoints[15][1];
	sourceY[44] = originFeaturePoints[14][1];
	sourceY[53] = originFeaturePoints[13][1];
	sourceY[62] = originFeaturePoints[12][1];
	sourceY[71] = originFeaturePoints[11][1];

	destX[1] = inputFeaturePoints[0][0];
	destX[2] = inputFeaturePoints[17][0];
	destX[3] = inputFeaturePoints[20][0];
	destX[4] = (inputFeaturePoints[21][0] + inputFeaturePoints[22][0]) / 2;
	destX[5] = inputFeaturePoints[23][0];
	destX[6] = inputFeaturePoints[26][0];
	destX[7] = inputFeaturePoints[16][0];

	destX[73] = inputFeaturePoints[4][0];
	destX[74] = inputFeaturePoints[5][0];
	destX[75] = inputFeaturePoints[6][0];
	destX[76] = inputFeaturePoints[8][0];
	destX[77] = inputFeaturePoints[10][0];
	destX[78] = inputFeaturePoints[11][0];
	destX[79] = inputFeaturePoints[12][0];

	destY[9] = inputFeaturePoints[17][1];
	destY[18] = inputFeaturePoints[0][1];
	destY[27] = inputFeaturePoints[1][1];
	destY[36] = inputFeaturePoints[2][1];
	destY[45] = inputFeaturePoints[3][1];
	destY[54] = inputFeaturePoints[4][1];
	destY[63] = inputFeaturePoints[5][1];

	destY[17] = inputFeaturePoints[26][1];
	destY[26] = inputFeaturePoints[16][1];
	destY[35] = inputFeaturePoints[15][1];
	destY[44] = inputFeaturePoints[14][1];
	destY[53] = inputFeaturePoints[13][1];
	destY[62] = inputFeaturePoints[12][1];
	destY[71] = inputFeaturePoints[11][1];




	sourceX[11] = originFeaturePoints[17][0];
	sourceY[11] = originFeaturePoints[17][1];

	sourceX[12] = originFeaturePoints[20][0];
	sourceY[12] = originFeaturePoints[20][1];

	sourceX[14] = originFeaturePoints[23][0];
	sourceY[14] = originFeaturePoints[23][1];

	sourceX[15] = originFeaturePoints[26][0];
	sourceY[15] = originFeaturePoints[26][1];

	sourceX[19] = originFeaturePoints[0][0];
	sourceY[19] = originFeaturePoints[0][1];

	sourceX[21] = originFeaturePoints[39][0];
	//sourceY[21] = (originFeaturePoints[38][1] + originFeaturePoints[40][1]) / 2;
	sourceY[21] = originFeaturePoints[39][1];

	sourceX[22] = originFeaturePoints[27][0];
	sourceY[22] = originFeaturePoints[27][1];

	sourceX[23] = originFeaturePoints[42][0];
	//sourceY[23] = (originFeaturePoints[43][1] + originFeaturePoints[47][1]) / 2;
	sourceY[23] = originFeaturePoints[42][1];

	sourceX[25] = originFeaturePoints[16][0];
	sourceY[25] = originFeaturePoints[16][1];

	sourceX[28] = originFeaturePoints[1][0];
	sourceY[28] = originFeaturePoints[1][1];

	sourceX[31] = originFeaturePoints[29][0];
	sourceY[31] = originFeaturePoints[29][1];

	sourceX[34] = originFeaturePoints[15][0];
	sourceY[34] = originFeaturePoints[15][1];

	sourceX[37] = originFeaturePoints[2][0];
	sourceY[37] = originFeaturePoints[2][1];

	sourceX[40] = originFeaturePoints[33][0];
	sourceY[40] = originFeaturePoints[33][1];

	sourceX[43] = originFeaturePoints[14][0];
	sourceY[43] = originFeaturePoints[14][1];

	sourceX[46] = originFeaturePoints[3][0];
	sourceY[46] = originFeaturePoints[3][1];

	sourceX[48] = originFeaturePoints[48][0];
	sourceY[48] = originFeaturePoints[48][1];

	sourceX[49] = (originFeaturePoints[62][0] + originFeaturePoints[66][0]) / 2;
	sourceY[49] = (originFeaturePoints[62][1] + originFeaturePoints[66][1]) / 2;

	sourceX[50] = originFeaturePoints[54][0];
	sourceY[50] = originFeaturePoints[54][1];

	sourceX[52] = originFeaturePoints[13][0];
	sourceY[52] = originFeaturePoints[13][1];

	sourceX[55] = originFeaturePoints[4][0];
	sourceY[55] = originFeaturePoints[4][1];

	sourceX[58] = originFeaturePoints[57][0];
	sourceY[58] = originFeaturePoints[57][1];

	sourceX[61] = originFeaturePoints[12][0];
	sourceY[61] = originFeaturePoints[12][1];

	sourceX[65] = originFeaturePoints[5][0];
	sourceY[65] = originFeaturePoints[5][1];

	sourceX[66] = originFeaturePoints[6][0];
	sourceY[66] = originFeaturePoints[6][1];

	sourceX[67] = originFeaturePoints[8][0];
	sourceY[67] = originFeaturePoints[8][1];

	sourceX[68] = originFeaturePoints[10][0];
	sourceY[68] = originFeaturePoints[10][1];

	sourceX[69] = originFeaturePoints[11][0];
	sourceY[69] = originFeaturePoints[11][1];

	destX[11] = inputFeaturePoints[17][0];
	destY[11] = inputFeaturePoints[17][1];

	destX[12] = inputFeaturePoints[20][0];
	destY[12] = inputFeaturePoints[20][1];

	destX[14] = inputFeaturePoints[23][0];
	destY[14] = inputFeaturePoints[23][1];

	destX[15] = inputFeaturePoints[26][0];
	destY[15] = inputFeaturePoints[26][1];

	destX[19] = inputFeaturePoints[0][0];
	destY[19] = inputFeaturePoints[0][1];

	destX[21] = inputFeaturePoints[39][0];
	destY[21] = inputFeaturePoints[39][1];

	destX[22] = inputFeaturePoints[27][0];
	destY[22] = inputFeaturePoints[27][1];

	destX[23] = inputFeaturePoints[42][0];
	destY[23] = inputFeaturePoints[42][1];

	destX[25] = inputFeaturePoints[16][0];
	destY[25] = inputFeaturePoints[16][1];

	destX[28] = inputFeaturePoints[1][0];
	destY[28] = inputFeaturePoints[1][1];

	destX[31] = inputFeaturePoints[29][0];
	destY[31] = inputFeaturePoints[29][1];

	destX[34] = inputFeaturePoints[15][0];
	destY[34] = inputFeaturePoints[15][1];

	destX[37] = inputFeaturePoints[2][0];
	destY[37] = inputFeaturePoints[2][1];

	destX[40] = inputFeaturePoints[33][0];
	destY[40] = inputFeaturePoints[33][1];

	destX[43] = inputFeaturePoints[14][0];
	destY[43] = inputFeaturePoints[14][1];

	destX[46] = inputFeaturePoints[3][0];
	destY[46] = inputFeaturePoints[3][1];

	destX[48] = inputFeaturePoints[48][0];
	destY[48] = inputFeaturePoints[48][1];

	destX[49] = (inputFeaturePoints[62][0] + inputFeaturePoints[66][0]) / 2;
	destY[49] = (inputFeaturePoints[62][1] + inputFeaturePoints[66][1]) / 2;

	destX[50] = inputFeaturePoints[54][0];
	destY[50] = inputFeaturePoints[54][1];

	destX[52] = inputFeaturePoints[13][0];
	destY[52] = inputFeaturePoints[13][1];

	destX[55] = inputFeaturePoints[4][0];
	destY[55] = inputFeaturePoints[4][1];

	destX[58] = inputFeaturePoints[57][0];
	destY[58] = inputFeaturePoints[57][1];

	destX[61] = inputFeaturePoints[12][0];
	destY[61] = inputFeaturePoints[12][1];

	destX[65] = inputFeaturePoints[5][0];
	destY[65] = inputFeaturePoints[5][1];

	destX[66] = inputFeaturePoints[6][0];
	destY[66] = inputFeaturePoints[6][1];

	destX[67] = inputFeaturePoints[8][0];
	destY[67] = inputFeaturePoints[8][1];

	destX[68] = inputFeaturePoints[10][0];
	destY[68] = inputFeaturePoints[10][1];

	destX[69] = inputFeaturePoints[11][0];
	destY[69] = inputFeaturePoints[11][1];

	/*
	sourceX[10] = originFeaturePoints[0][0];
	sourceX[16] = originFeaturePoints[16][0]; 
	sourceX[64] = originFeaturePoints[4][0];
	sourceX[70] = originFeaturePoints[12][0];

	destX[10] = inputFeaturePoints[0][0];
	destX[16] = inputFeaturePoints[16][0];
	destX[64] = inputFeaturePoints[4][0];
	destX[70] = inputFeaturePoints[12][0];
	*/
#pragma endregion createMesh

	for (int i = 10; i < 71; i++)
	{
		if (sourceX[i] == -1) {
			if (sourceX[i + 9] == -1){
				sourceX[i] = sourceX[i - 9];
				destX[i] = destX[i - 9];
			}
			else{
				sourceX[i] = (sourceX[i - 9] + sourceX[i + 9]) / 2;
				destX[i] = (destX[i - 9] + destX[i + 9]) / 2;
			}
		}
	}

	/*
	for (int i = 10; i < 71; i++)
	{
		if (sourceX[i] == -1) {
			if (sourceX[i + 1] == -1){
				sourceX[i] = (2 * sourceX[i - 1] + sourceX[i + 2]) / 3;
				sourceX[i + 1] = (sourceX[i - 1] + 2 * sourceX[i + 2]) / 3;
			}
			else{
				sourceX[i] = (sourceX[i - 1] + sourceX[i + 1]) / 2;
			}
		}
	}
	for (int i = 10; i < 71; i++)
	{
		if (destX[i] == -1) {
			if (destX[i + 1] == -1){
				destX[i] = (2 * destX[i - 1] + destX[i + 2]) / 3;
				destX[i + 1] = (destX[i - 1] + 2 * destX[i + 2]) / 3;
			}
			else{
				destX[i] = (destX[i - 1] + destX[i + 1]) / 2;
			}
		}
	}
	*/
	for (int i = 10; i < 71; i++)
	{
		if (sourceY[i] == -1) {
			if (sourceY[i + 1] == -1){
				sourceY[i] = (2 * sourceY[i - 1] + sourceY[i + 2]) / 3;
				sourceY[i + 1] = (sourceY[i - 1] + 2 * sourceY[i + 2]) / 3;
			}
			else{
				sourceY[i] = (sourceY[i - 1] + sourceY[i + 1]) / 2;
			}
		}
	}

	for (int i = 10; i < 71; i++)
	{
		if (destY[i] == -1) {
			if (destY[i + 1] == -1){
				destY[i] = (2 * destY[i - 1] + destY[i + 2]) / 3;
				destY[i + 1] = (destY[i - 1] + 2 * destY[i + 2]) / 3;
			}
			else{
				destY[i] = (destY[i - 1] + destY[i + 1]) / 2;
			}
		}
	}

	for (int i = 0; i < 81; i++){
		if (i % 9 == 0) printf("\n");
		printf("%d|%d	", sourceX[i], destX[i]);
	}
	printf("\n");
	for (int i = 0; i < 81; i++){
		if (i % 9 == 0) printf("\n");
		printf("%d|%d	", sourceY[i], destY[i]);
	}
 	MeshWarp(bytes, image.size().height, image.size().width, 9, 9, sourceX, sourceY, destX, destY, newbytes);

	frontalImage = Mat(image.size().height, image.size().width, CV_8UC3, newbytes).clone();
	
	if (!frontalImage.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("result1", WINDOW_AUTOSIZE);
	imshow("result1", frontalImage);
	namedWindow("origin", WINDOW_AUTOSIZE);
	imshow("origin", image);
	waitKey(0);
	return EXIT_SUCCESS;

}