
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "data.hpp"

using namespace std;

int Data::loadData(const char *location) {

	FILE *file;
	
	file = fopen(location, "r");

	clog << "\nParsing data :";

	if (file == NULL) {
		clog << "\nFile" << location << " not found.";
		return -1;
	}

	length = 0;	

	char str[100];
	while (fgets( str, 100, file) != NULL) {
		length++;
	}

	clog << "\nRead " << length << " items.";

	x = new int[length];
	y = new int[length];
	z = new int[length];

	rx = new float[length];
	ry = new float[length];
	rz = new float[length];
	
	rewind(file);

	int i1,i2,i3;
	float f1,f2,f3;
	int i = 0;
	while (fscanf(file, "%i %i %i %f %f %f", &i1, &i2, &i3, &f1, &f2, &f3) == 6) {
		
		x[i] = i1;
		y[i] = i2;
		z[i] = i3;

		rx[i] = f1;
		ry[i] = f2;
		rz[i] = f3;

		i++;
	}

	return 0;
}
		
	void Data::printData() {	
		
		for (int i = 0; i < length; i++) {
			cout << "\n\t" << x[i] << "\t" << y[i] << "\t" << z[i] << "\t" << rx[i] << "\t" << ry[i] << "\t" << rz[i];
		}
		cout << "\n";
	}
