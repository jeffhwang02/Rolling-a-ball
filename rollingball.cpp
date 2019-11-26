#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/nonfree/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <immintrin.h>
#include <cstdint>
#include <chrono>
#include <thread>
#include <winsock2.h>
#include <algorithm>

using namespace cv;
using namespace std;
#define SQR(x)  ((x)*(x))

struct RollingBall {
	std::vector<float> data;
	int patchwidth;
	int shrinkfactor;
	void Rollingball(int radius) {
		int artimper=0;
		if (radius <= 10) {
			shrinkfactor = 1;
			artimper = 24;
		}
		else if (radius <= 30) {
			shrinkfactor = 2;
			artimper = 24;
		}
		else if (radius <= 100) {
			shrinkfactor = 4;
			artimper = 32;
		}
		else {
			shrinkfactor = 8;
			artimper = 40;
		}

		Build(radius, artimper);

	}

	void Build(int ballradius, int artimper) {
		int sballradius = ballradius / shrinkfactor;
		if (sballradius < 1)
			sballradius = 1;
		int rsquare = SQR(sballradius);
		int diam = sballradius * 2;
		int xtrim = int((artimper* sballradius) / 100);
		//patchwidth = round(sballradius - xtrim ) ;
		int halfpatchwidth = (sballradius - xtrim);
		patchwidth = 2*halfpatchwidth+1;
		int ballsize = patchwidth* patchwidth;
		data.resize(ballsize);
		int p = 0;
		for (int y = 0; y < patchwidth; y++)
		{
			for (int x = 0; x < patchwidth; x++)
			{


				int xval = x - halfpatchwidth;
				int yval = y - halfpatchwidth;
				float temp = (float)rsquare - (float)SQR(xval) - (float)SQR(yval);
				if (temp >= 0)
					data[p] = float(sqrt(temp));
				else
					data[p] = 0;
				p++;
			}
		}

	}

};


void RollBall(RollingBall &ball, Mat src, int swidth, int sheight, Mat sbackground) {

	Mat src2;
	src.convertTo(src2, CV_32FC1);
	src2 = 255 - src2;
	vector<float> float_img;
	for (int j= 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			float_img.push_back(src2.at<float>(j, i));
		}

	}
	/////////////////////////////////////
	vector<float> z_ball = ball.data;
	int	ball_width = ball.patchwidth;
	int	radius = int(ball_width / 2);
	vector<float>	cache;
	cache.resize(swidth * ball_width);
	
	for (int y = -radius; y < sheight + radius; y++) {
		int next_line_to_write = (y + radius) % ball_width;
		int next_line_to_read = y + radius;
		if (next_line_to_read < sheight) {
			int src = next_line_to_read * swidth;
			int dest = next_line_to_write * swidth;
			//float inf = float(-INT_MAX);
			memcpy(&cache[dest], &float_img[src], 4*swidth);
			//memcpy(&float_img[src], &inf, 4 * swidth);
			for (int i = src; i < src+swidth; i++)
			{
				float_img[i] = float(-INT_MAX);
			}
		}

		int y0 = MAX((0), (y - radius));
		int y_ball0 = y0 - y + radius;
		int y_end = y + radius;
		if (y_end >= sheight)
			y_end = sheight - 1;
		for (int x = -radius; x < swidth + radius; x++) {
			float z = float(INT_MAX);
			int x0 = MAX((0), (x - radius));
			int x_ball0 = x0 - x + radius;
			int x_end = x + radius;
			if (x_end >= swidth)
				x_end = swidth - 1;

			int y_ball = y_ball0;
			for (int yp = y0; yp < y_end + 1; yp++) {
				int cache_pointer = (yp % ball_width) * swidth + x0;
				int bp = x_ball0 + y_ball * ball_width;
				for (int xp = x0; xp < x_end + 1; xp++) {
					float z_reduced = cache[cache_pointer] - z_ball[bp];
					if (z > z_reduced)
						z = z_reduced;
					cache_pointer += 1;
					bp += 1;
				}
				y_ball += 1;
			}

			y_ball = y_ball0;
			for (int yp = y0; yp < y_end + 1; yp++) {
				int p = x0 + yp * swidth;
				int bp = x_ball0 + y_ball * ball_width;
				for (int xp = x0; xp < x_end + 1; xp++) {
					float z_min = z + z_ball[bp];
					if (float_img[p] < z_min)
						float_img[p] = z_min;
					p += 1;
					bp += 1;
				}
				y_ball += 1;
			}

		}

	}


	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			//float_img.push_back(src2.at<float>(j, i));
			sbackground.at<float>(j, i) = float_img[j*src.cols + i];
		}

	}
};


Mat RollingBallMat(Mat src, int radius) {
	Mat dst2 = Mat::zeros(src.size(), CV_32FC1);
	if (src.type() != CV_8UC1)
		return dst2;
	if (src.empty())
		return dst2;

	RollingBall ball;
	ball.Rollingball(radius);
	//Mat src2 = Mat::zeros(src.size(), CV_32FC1);
	//src.convertTo(src2, CV_32FC1);
	//src2 = src2 / 256;
	RollBall(ball, src, src.cols, src.rows, dst2);
	
	return dst2;
}


int main(int argc, char** argv) {
	//Mat src = imread("C:/Users/LGPC/Downloads/opencv-rolling-ball-master/opencv-rolling-ball-master/outputs/example.png", 0);
	Mat src = imread("C:/Users/LGPC/Downloads/opencv-rolling-ball-master/opencv-rolling-ball-master/outputs/example.png", 0);
	resize(src, src, Size(0, 0), 0.25, 0.25);
	
	Mat dst, back;
	dst =RollingBallMat(src,  30);
	//NonUniformIlluminationMorph(src, dst, 7, true, &back);
	Mat equalM;
	equalizeHist(src, equalM);
	imshow("src", src);
	double min, max;
	cv::minMaxLoc(dst, &min, &max);
	//imshow("dst", dst);
	//minMaxLoc(dst, double(0.0), double(255.0));
	//dst = (255-dst);
	dst.convertTo(dst, CV_8UC1);
	imshow("dst2", dst);
	imshow("dst3", src-dst);
	//imshow("equalM", equalM);
	waitKey();
}

