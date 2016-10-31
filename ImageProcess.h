// ImageProcess.h : main header file for the ImageProcess DLL
//

#pragma once

#ifndef __AFXWIN_H__
//#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols

#define BYTE unsigned char
#define	PI		3.1415926535898
#define MAX(a,b)	(((a) > (b)) ? (a) : (b))
#define MIN(a,b)	(((a) < (b)) ? (a) : (b))

template<typename T>
inline T limit(const T& value)
{
	return ((value > 255) ? 255 : ((value < 0) ? 0 : value));
}

struct hsv_color {
	double h;        // Hue
	double s;        // Saturation
	double v;        // Value
};

//BYTE* matToBytes(Mat image);
//Mat bytesToMat(BYTE* bytes, int width, int height);


//Warping
//extern "C" __declspec(dllexport)
void MeshWarp(BYTE* srcImage, int height, int width, int numMeshH, int numMeshV, int* sourceX, int* sourceY, int* destX, int* destY, BYTE* outImage);
bool InterpolSpline(double *x1, double *y1, int len1, double *x2, double *y2, int len2);
void Resample(double f[], BYTE* in, BYTE* out, int INlen, int OUTlen);


//Preprocess
void ExtractVertex(BYTE* srcArray, int height, int width, BYTE* retArray, int vertexType, BYTE threshold);

void ColoredBinarization(BYTE* srcArray, int height, int width, BYTE* retArray, int vertexType);
hsv_color RGB2HSV(BYTE r, BYTE g, BYTE b);

void GetVertexPos(BYTE* srcArray, int height, int width, int* points);
void Labeling(BYTE* srcArray, int height, int width, int* arrLabel, int &nArr);
void DepthFirstSearch(BYTE* srcimg, int height, int width, int pX, int pY, int &nPoint, int &x_range, int &y_range);

void CropImage(BYTE* srcArray, int height, int width, int px, int py, int cropWidth, int cropHeight, BYTE* retArray);

void GetStampPos(BYTE* srcArray, int height, int width, int* rect, BYTE threshold);

void PerspectiveMapping(BYTE* srcArray, int srcHeight, int srcWidth, int margin, double* srcX, double* srcY, BYTE* retArray, int retHeight, int retWidth);