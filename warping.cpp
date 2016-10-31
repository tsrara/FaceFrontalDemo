
#include "stdlib.h"
#include "string.h"
#include "ImageProcess.h"
//#include "opencv2/core/core.hpp"
//#include "opencv2/opencv.hpp"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/*
using cv::Mat;

BYTE* matToBytes(Mat image)
{
int size = image.total() * image.elemSize();
BYTE * bytes = new BYTE[size];  // you will have to delete[] that later
std::memcpy(bytes, image.data, size * sizeof(BYTE));
}

Mat bytesToMat(BYTE* bytes, int width, int height)
{
Mat image = Mat(height, width, CV_8UC3, bytes).clone(); // make a copy
//return image;
}
*/

void MeshWarp(BYTE* srcImage, int height, int width, int numMeshH, int numMeshV, int* sourceX, int* sourceY, int* destX, int* destY, BYTE* outImage)
{
	// 주의사항
	//
	// 본 함수에서 사용되는 parameter 중 Mesh의 좌표는 모든 점의 좌표를 포함해야 합니다.
	// 예를들어 5 x 5 Mesh에서 사용될 점 좌표의 배열의 길이는 각각 25가 됩니다.

	int x, y;
	BYTE* srcLineW = (BYTE *)malloc(width * 3 * sizeof(BYTE));
	BYTE* dstLineW = (BYTE *)malloc(width * 3 * sizeof(BYTE));
	BYTE* srcLineH = (BYTE *)malloc(height * 3 * sizeof(BYTE));
	BYTE* dstLineH = (BYTE *)malloc(height * 3 * sizeof(BYTE));
	BYTE* tmpimg = (BYTE *)malloc(height * width * 3 * sizeof(BYTE));
	int   maxLen = MAX(height, width) + 1;

	double *pIndex = (double *)malloc(maxLen * sizeof(double));
	double *pRowX1 = (double *)malloc(maxLen * sizeof(double)); // source mesh
	double *pRowY1 = (double *)malloc(maxLen * sizeof(double)); // 
	double *pRowX2 = (double *)malloc(maxLen * sizeof(double)); // destination mesh
	double *pRowY2 = (double *)malloc(maxLen * sizeof(double)); // 
	double *pMap1 = (double *)malloc(maxLen * sizeof(double)); // interpolated spline
	double *pMap2 = (double *)malloc(maxLen * sizeof(double)); // interpolated spline
	double *pMap3 = (double *)malloc(maxLen * sizeof(double)); // interpolated spline

	memset(pMap1, 0, maxLen * sizeof(double));
	memset(pMap2, 0, maxLen * sizeof(double));
	memset(pMap3, 0, maxLen * sizeof(double));

	//// prepare first pass
	// souce의 vertical spline에 대해 x-intercept table을 생성
	double *pSrcMeshTab = (double *)malloc(numMeshH * height * sizeof(double));
	double *pDstMeshTab = (double *)malloc(numMeshH * height * sizeof(double));

	for (y = 0; y < height; y++)
		pIndex[y] = y;

	// get vertical spline
	for (x = 0; x < numMeshH; x++)
	{
		for (y = 0; y < numMeshV; y++)
		{
			pRowX1[y] = (double)sourceX[y * numMeshH + x];
			pRowY1[y] = (double)sourceY[y * numMeshH + x];

			pRowX2[y] = (double)destX[y * numMeshH + x];
			pRowY2[y] = (double)destY[y * numMeshH + x];
		}

		InterpolSpline(pRowY1, pRowX1, numMeshV, pIndex, pMap1, height);
		InterpolSpline(pRowY2, pRowX2, numMeshV, pIndex, pMap2, height);

		for (y = 0; y < height; y++)
		{
			pSrcMeshTab[numMeshH * y + x] = pMap1[y];
			pDstMeshTab[numMeshH * y + x] = pMap2[y];
		}
	}

	//// first pass
	for (x = 0; x < width; x++)
		pIndex[x] = x;

	for (y = 0; y < height; y++)
	{
		double *pX1 = &pSrcMeshTab[numMeshH * y];
		double *pX2 = &pDstMeshTab[numMeshH * y];

		InterpolSpline(pX1, pX2, numMeshH, pIndex, pMap1, width);

		for (x = 0; x < width; x++)
		{
			srcLineW[x * 3 + 2] = srcImage[y * width * 3 + x * 3 + 2];
			srcLineW[x * 3 + 1] = srcImage[y * width * 3 + x * 3 + 1];
			srcLineW[x * 3] = srcImage[y * width * 3 + x * 3];
		}

		Resample(pMap1, srcLineW, dstLineW, width, width);

		for (x = 0; x < width; x++)
		{
			tmpimg[y * width * 3 + x * 3 + 2] = dstLineW[x * 3 + 2];
			tmpimg[y * width * 3 + x * 3 + 1] = dstLineW[x * 3 + 1];
			tmpimg[y * width * 3 + x * 3] = dstLineW[x * 3];
		}
	}

	//// prepare second pass
	double* pIndex_h = (double *)malloc(maxLen*sizeof(double));
	double* pRowX1_h = (double *)malloc(maxLen*sizeof(double));
	double* pRowY1_h = (double *)malloc(maxLen*sizeof(double));
	double* pRowX2_h = (double *)malloc(maxLen*sizeof(double));
	double* pRowY2_h = (double *)malloc(maxLen*sizeof(double));
	double* pSrcMeshTab_h = (double *)malloc(numMeshV * width * sizeof(double));
	double* pDstMeshTab_h = (double *)malloc(numMeshV * width * sizeof(double));

	for (x = 0; x < width; x++)
		pIndex_h[x] = x;

	// get horizontal spline
	for (y = 0; y < numMeshV; y++)
	{
		for (x = 0; x < numMeshH; x++)
		{
			pRowX1_h[x] = (double)sourceX[y * numMeshH + x];
			pRowY1_h[x] = (double)sourceY[y * numMeshH + x];

			pRowX2_h[x] = (double)destX[y * numMeshH + x];
			pRowY2_h[x] = (double)destY[y * numMeshH + x];
		}

		InterpolSpline(pRowX1_h, pRowY1_h, numMeshH, pIndex_h, &pSrcMeshTab_h[width * y], width);
		InterpolSpline(pRowX2_h, pRowY2_h, numMeshH, pIndex_h, &pDstMeshTab_h[width * y], width);
	}

	//// second pass

	for (y = 0; y < height; y++)
		pIndex_h[y] = y;

	for (x = 0; x < width; x++)
	{
		for (y = 0; y < numMeshV; y++)
		{
			pRowX1_h[y] = pSrcMeshTab_h[y *	width + x];
			pRowY1_h[y] = pDstMeshTab_h[y * width + x];
		}

		InterpolSpline(pRowX1_h, pRowY1_h, numMeshV, pIndex_h, pMap1, height);

		for (y = 0; y < height; y++)
		{
			srcLineH[y * 3 + 2] = tmpimg[y * width * 3 + x * 3 + 2];
			srcLineH[y * 3 + 1] = tmpimg[y * width * 3 + x * 3 + 1];
			srcLineH[y * 3] = tmpimg[y * width * 3 + x * 3];
		}

		Resample(pMap1, srcLineH, dstLineH, height, height);

		for (y = 0; y < height; y++)
		{
			outImage[y * width * 3 + x * 3 + 2] = dstLineH[y * 3 + 2];
			outImage[y * width * 3 + x * 3 + 1] = dstLineH[y * 3 + 1];
			outImage[y * width * 3 + x * 3] = dstLineH[y * 3];
		}
	}

	free(tmpimg);

	free(srcLineW);
	free(dstLineW);
	free(srcLineH);
	free(dstLineH);

	free(pSrcMeshTab);
	free(pDstMeshTab);

	free(pIndex);
	free(pRowX1);
	free(pRowY1);
	free(pRowX2);
	free(pRowY2);
	free(pMap1);
	free(pMap2);
	free(pMap3);

	free(pIndex_h);
	free(pRowX1_h);
	free(pRowY1_h);
	free(pRowX2_h);
	free(pRowY2_h);
	free(pSrcMeshTab_h);
	free(pDstMeshTab_h);
}

///////////////////////////////////////////////////////////////////
// Function	   : InterpolSpline
// Description : Spline을 구하는 함수
// Parameter   : (Point의 x좌표, y좌표, 길이) X2
// Return	   : TRUE(성공)
///////////////////////////////////////////////////////////////////
bool InterpolSpline(double *x1, double *y1, int len1, double *x2, double *y2, int len2)
{
	int     i, j, dir, j1, j2;
	double  p1, p2, p3;
	double  x, dx1, dx2;
	double  dx, dy, yd1, yd2;
	double  a0y, a1y, a2y, a3y;

	if (x1[0] < x1[1])
	{	// 증가 
		if (x2[0] < x1[0] || x2[len2 - 1]>x1[len1 - 1]) dir = 0;
		else dir = 1;
	}
	else
	{	// 감소 
		if (x2[0] > x1[0] || x2[len2 - 1] < x1[len1 - 1]) dir = 0;
		else dir = -1;
	}

	if (dir == 0)
	{
		return false;
	}

	// p1 : 인터벌의 첫번째 endpoint
	// p2 : resampling 위치
	// p3 : 인터벌의 두번째 endpoint
	// j  : is input index for current interval
	//

	// coefficient 초기화
	//
	if (dir == 1)
		p3 = x2[0] - 1;
	else
		p3 = x2[0] + 1;

	for (i = 0; i < len2; i++)
	{
		// 새로운 인터벌의 시작인가
		p2 = x2[i];
		if ((dir == 1 && p2 > p3) || (dir == -1 && p2 < p3))
		{
			// p2 를 포함하는 인터벌을 찾는다.
			if (dir)
			{
				for (j = 0; j < len1 && p2 > x1[j]; j++);
				if (p2 < x1[j]) j--;
			}
			else
			{
				for (j = 0; j < len1 && p2 < x1[j]; j++);
				if (p2 > x1[j]) j--;
			}

			p1 = x1[j];			// 첫번째 endpt 수정
			p3 = x1[j + 1];		// 두번째 endpt 수정

			j1 = max(j - 1, 0);
			j2 = min(j + 2, len1 - 1);

			if (p3 == p1) dx = 1.0;
			else		  dx = 1.0 / (p3 - p1);
			if (p3 == x1[j1]) dx1 = 1.0;
			else			  dx1 = 1.0 / (p3 - x1[j1]);
			if (x1[j2] == p1) dx2 = 1.0;
			else			  dx2 = 1.0 / (x1[j2] - p1);

			dy = (y1[j + 1] - y1[j])  * dx;
			yd1 = (y1[j + 1] - y1[j1]) * dx1;
			yd2 = (y1[j2] - y1[j])  * dx2;
			a0y = y1[j];
			a1y = yd1;
			a2y = dx * (3 * dy - 2 * yd1 - yd2);
			a3y = dx * dx * (-2 * dy + yd1 + yd2);
		}

		x = p2 - p1;
		y2[i] = ((a3y * x + a2y) * x + a1y) * x + a0y;
	}

	return true;
}

///////////////////////////////////////////////////////////////////
// Function	   : Resample
// Description : Fant 알고리즘에 기반한 재추출을 하는 함수
// Parameter   : Mesh array, 소스 image line , 결과 image line, 소스 길이, 결과 길이
// Return	   : void
///////////////////////////////////////////////////////////////////
void Resample(double f[], BYTE* in, BYTE* out, int INlen, int OUTlen)
{
	int u, x, index1, index2;
	BYTE v0, v1;
	double acc, intensity, SIZFAC, INSEG, OUTSEG;
	double* inpos = (double *)malloc(OUTlen * sizeof(double));

	// 각 출력 픽셀에 대한 입력 인덱스를 미리 계산
	u = 0;

	for (x = 0; x < OUTlen; x++)
	{
		while (f[u + 1] < x)
			u++;
		inpos[x] = u + (double)(x - f[u]) / (f[u + 1] - f[u]);
	}

	// RED 채널 에 대해 
	//////////////////////////////////////////////////////////////////////
	INSEG = 1.0;              // 초기치 : 입력픽셀 한개가 전부 쓰임
	OUTSEG = inpos[1];        // 1개의 출력 픽셀에 대한 입력 픽셀의 수
	SIZFAC = OUTSEG;          // 1/스케일 팩터
	acc = 0.0;                // 누적값

	// 모든 출력 픽셀에 대해 계산
	index1 = 0;
	index2 = 0;
	u = 0;

	v0 = in[index1];
	index1 += 3;
	v1 = in[index1];
	index1 += 3;

	for (u = 1; u < OUTlen;)
	{
		intensity = INSEG * v0 + (1 - INSEG) * v1;

		if (INSEG < OUTSEG) {
			acc += (intensity * INSEG);
			OUTSEG -= INSEG;
			INSEG = 1.0;
			v0 = v1;
			v1 = in[index1];
			index1 += 3;
		}
		else {
			acc += (intensity * OUTSEG);
			acc /= SIZFAC;
			out[index2] = (BYTE)min(acc, 255);
			index2 += 3;
			acc = 0.0;
			INSEG -= OUTSEG;
			OUTSEG = inpos[u + 1] - inpos[u];
			SIZFAC = OUTSEG;
			u++;
		}
	}

	// GREEN 채널 에 대해
	//////////////////////////////////////////////////////////////////////
	INSEG = 1.0;              // 초기치 : 입력픽셀 한개가 전부 쓰임
	OUTSEG = inpos[1];        // 1개의 출력 픽셀에 대한 입력 픽셀의 수
	SIZFAC = OUTSEG;          // 1/스케일 팩터
	acc = 0.0;                // 누적값

	// 모든 출력 픽셀에 대해 계산
	index1 = 1;
	index2 = 1;
	u = 0;

	v0 = in[index1];
	index1 += 3;
	v1 = in[index1];
	index1 += 3;

	for (u = 1; u < OUTlen;)
	{
		intensity = INSEG * v0 + (1 - INSEG) * v1;

		if (INSEG < OUTSEG) {
			acc += (intensity * INSEG);
			OUTSEG -= INSEG;
			INSEG = 1.0;
			v0 = v1;
			v1 = in[index1];
			index1 += 3;
		}
		else {
			acc += (intensity * OUTSEG);
			acc /= SIZFAC;
			out[index2] = (BYTE)min(acc, 255);
			index2 += 3;
			acc = 0.0;
			INSEG -= OUTSEG;
			OUTSEG = inpos[u + 1] - inpos[u];
			SIZFAC = OUTSEG;
			u++;
		}
	}

	// BLUE 채널 에 대해
	//////////////////////////////////////////////////////////////////////
	INSEG = 1.0;              // 초기치 : 입력픽셀 한개가 전부 쓰임
	OUTSEG = inpos[1];        // 1개의 출력 픽셀에 대한 입력 픽셀의 수
	SIZFAC = OUTSEG;          // 1/스케일 팩터
	acc = 0.0;                // 누적값

	// 모든 출력 픽셀에 대해 계산
	index1 = 2;
	index2 = 2;
	u = 0;

	v0 = in[index1];
	index1 += 3;
	v1 = in[index1];
	index1 += 3;

	for (u = 1; u < OUTlen;)
	{
		intensity = INSEG * v0 + (1 - INSEG) * v1;

		if (INSEG < OUTSEG) {
			acc += (intensity * INSEG);
			OUTSEG -= INSEG;
			INSEG = 1.0;
			v0 = v1;
			v1 = in[index1];
			index1 += 3;
		}
		else {
			acc += (intensity * OUTSEG);
			acc /= SIZFAC;
			out[index2] = (BYTE)min(acc, 255);
			index2 += 3;
			acc = 0.0;
			INSEG -= OUTSEG;
			OUTSEG = inpos[u + 1] - inpos[u];
			SIZFAC = OUTSEG;
			u++;
		}
	}

	free(inpos);
}