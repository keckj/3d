

#include "image.hpp"

#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

void Image::reverseImage(const Mat& src, Mat& dst) {
	
	assert(src.rows == dst.rows && src.cols == dst.cols);

	for (int i = 0; i < src.rows; i++) {
		src.row(i).copyTo(dst.row(dst.rows - i - 1));
	}
}
