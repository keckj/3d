#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <list>

using namespace std;
using namespace cv;

class Image {
	public:
		
		static void reverseImage(const Mat& src, Mat& dst);
};
