

#include "image.hpp"
#include "log.hpp"

#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

vector<string> *listdir(const char *dirname) {
	DIR *dp;
	dirent *d;
	vector<string> *vec = new vector<string>;

	dp = opendir(dirname);
	while((d = readdir(dp)) != NULL)
		vec->push_back(d->d_name);

	sort(vec->begin(), vec->end());
	return vec;
}

int Image::loadImage(const char *name) {

	Mat *image = new Mat();

	*image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);

	if(! image->data) {
		log_console.warn("Failed to load image %s .", name);
		return -1;
	}

	images.push_back(image);

	return 0;
}

int Image::loadImageFolder(const char* folder) {
	
	vector<string> *files = listdir(folder);
	
	char str[200];
	int nbImg = 0;

	for (vector<string>::iterator it = files->begin(); it != files->end(); it++) {
		if (*it != "." && *it != "..") {
			sprintf(str, "%s%s", folder, it->c_str());
			nbImg += (loadImage(str) + 1);
		}
	}

	log_console.info("Loaded %i images from folder %s", nbImg, folder);
	
	return nbImg;
}

void Image::computeGradientVectorFlow() {

	log_console.info("Computing Gradient Vector Flow");
	
	Mat *img = images.front();
	
	log_console.debug("Loaded image");
	log_console.debug("Size : %i x %i", img->size[0], img->size[1]);
	log_console.debug("Step : %i x %i", img->step[0], img->step[1]);
	

	//CANNY
	Mat gaussianBlurredImg, canny;
	log_console.debug("Applying Canny filter");
	GaussianBlur(*img, gaussianBlurredImg, Size(9,9), 5.0);
	int lowThreshold = 30;
	int kernel_size = 3;
	Canny(gaussianBlurredImg, canny, lowThreshold, lowThreshold*3, kernel_size);
	
	canny.copyTo(*img, canny);
	displayImage(*img);

	//THRESHOLD
	Mat thres;
	log_console.debug("Thresholding");
	threshold(canny, thres, 127, 255, THRESH_BINARY_INV);
	displayImage(thres);
	
	//SOBEL
	log_console.debug("Applying Sobel filter");
	Mat f(thres.size[0], thres.size[1], CV_16S, Scalar::all(0));
	Mat fx, fy, b, c1, c2;
	
	//add(f, 0, thres, f);
	displayImage(f);

	Sobel(f, fx, CV_16S, 1, 0, 3);
	Sobel(f, fy, CV_16S, 0, 1, 3);
	
	//convertScaleAbs(sobelY, sobelY);
	//convertScaleAbs(sobelX, sobelX);
	//addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);

}

void Image::displayImage(Mat &m) {

	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
	imshow( "Display window", m); 
	waitKey(0);

}
		

void Image::reverseImage(const Mat& src, Mat& dst) {
	
	assert(src.rows == dst.rows && src.cols == dst.cols);

	for (int i = 0; i < src.rows; i++) {
		src.row(i).copyTo(dst.row(dst.rows - i - 1));
	}
}
