#pragma once

#include <iostream>

using namespace std;

class Data {

	public:
		int length;

		int *x, *y, *z;
		float *rx, *ry, *rz;

		int loadData(const char *location);
		void filterData();
	
		void printData();
};
		
