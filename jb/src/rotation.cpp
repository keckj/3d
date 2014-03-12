
#include "rotation.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

using namespace std;

const char* eulerOrderString[] = { "XZX", "XYX", "YXY", "YZY", "ZYZ", "ZXZ"}; 
const char* taitBryanOrderString[] = { "XYZ", "YZX", "ZXY", "ZYX", "YXZ", "XZY"}; 
const char* orientationTypeString[] = { "Euler", "Tait-Bryan", "Quaternion" };
const char* rotationTypeString[] = { "Intrinsic", "Extrinsic" };

const char* toStringRotationType(RotationType r) {
	return rotationTypeString[r];	
}

const char* toStringOrientationType(OrientationType o) { 
	return orientationTypeString[o]; 
}

const char* toStringEulerOrder(EulerOrder e) {
	return eulerOrderString[e]; 
}

const char* toStringTaitBryanOrder(TaitBryanOrder e) { 
	return taitBryanOrderString[e]; 
}

const char* toStringAxeOrder(Order o, OrientationType t) {
	switch(t) {
		case(EULER_ORIENTATION): 
			return toStringEulerOrder(o.euler); 
			break;
		case(TAIT_BRYAN_ORIENTATION): 
			return toStringTaitBryanOrder(o.tb); 
			break;
		case(QUATERNION_ORIENTATION):
			return "XYZR";
			break;
		default:
			return "NO ORDER";
	}
}

void Rotation::initializeRotation(RotationType rotationType, OrientationType orientationType, Order axeOrder, float *orientationData) {
	Rotation::rotationType = rotationType;
	Rotation::orientationType = orientationType;
	Rotation::axeOrder = axeOrder;
	
	int nData = 3;
	nData += (orientationType == QUATERNION_ORIENTATION);

	if(orientationData != NULL) {
		this->orientationData = new float[nData];
		for(int i = 0; i < nData; i++) 
			this->orientationData[i] = orientationData[i];
	}
	else {
		Rotation::orientationData = NULL;
	}
}

Rotation::Rotation() {
	orientationData = NULL;
}

Rotation::~Rotation() {
	delete [] orientationData;
}

Rotation::Rotation(const Rotation& other) {
	initializeRotation(other.rotationType, other.orientationType, other.axeOrder, other.orientationData);
}

Rotation& Rotation::operator= (const Rotation& other) {
	Rotation temp(other);

	this->rotationType = other.rotationType;
	this->orientationType = other.orientationType;
	this->axeOrder = other.axeOrder;

	std::swap(temp.orientationData, this->orientationData);

	return *this;
}


Rotation::Rotation(RotationType rotationType, OrientationType orientationType, EulerOrder eulerOrder, float *eulerAngles) {
	Order axeOrder;
	axeOrder.euler = eulerOrder;

	initializeRotation(rotationType, orientationType, axeOrder, eulerAngles);
}


Rotation::Rotation(RotationType rotationType, OrientationType orientationType, TaitBryanOrder taitBryanOrder, float *cardanAngles) {
	Order axeOrder;
	axeOrder.tb = taitBryanOrder;

	initializeRotation(rotationType, orientationType, axeOrder, cardanAngles);
}

string Rotation::toString() {
	
	stringstream s("");

	s << "\n-- Rotation --"
	  << "\n\tRotation type : " 
	  << toStringRotationType(rotationType)
	  << "\n\tOrientation type : "
	  << toStringOrientationType(orientationType)
	  << "\n\tAxe order : " 
	  << toStringAxeOrder(axeOrder, orientationType)
	  << "\n\tOrientation data :  "
	  << orientationData[0] << "  " << orientationData[1] << "  " << orientationData[2]
	  << "\n";

	return s.str();
}


float *Rotation::getRotationMatrix() {

	float *rotationMatrix = new float[9]; 	

	switch(rotationType) 
	{
		case(QUATERNION_ORIENTATION):
			break;

		case(TAIT_BRYAN_ORIENTATION): 
		{	
			const float yaw = orientationData[0];
			const float pitch = orientationData[1];
			const float roll = orientationData[2];
			
			const float Sx = sinf(yaw);
			const float Sy = sinf(pitch);
			const float Sz = sinf(roll);
			const float Cx = cosf(yaw);
			const float Cy = cosf(pitch);
			const float Cz = cosf(roll);
			
			switch(axeOrder.tb) 
			{
				case ORDER_XYZ:
					rotationMatrix[0] = Cx*Cy;
					rotationMatrix[1] = Cx*Sy*Sz - Cz*Sx;
					rotationMatrix[2] = Sx*Sz + Cx*Cz*Sy;
					rotationMatrix[3] = Cy*Sx;
					rotationMatrix[4] = Cx*Cz + Sx*Sy*Sz;
					rotationMatrix[5] = Cz*Sx*Sy - Cx*Cz;
					rotationMatrix[6] = -Sy;
					rotationMatrix[7] = Cy*Sz;
					rotationMatrix[8] = Cy*Cz;
					break;

				case ORDER_YZX:
					rotationMatrix[0] = Cy*Cz;
					rotationMatrix[1] = Sx*Sy-Cx*Cy*Sz;
					rotationMatrix[2] = Cx*Sy+Cy*Sx*Sz;
					rotationMatrix[3] = Sz;
					rotationMatrix[4] = Cx*Cz;
					rotationMatrix[5] = -Cz*Sx;
					rotationMatrix[6] = -Cz*Sy;
					rotationMatrix[7] = Cy*Sx+Cx*Sy*Sz;
					rotationMatrix[8] = Cx*Cy-Sx*Sy*Sz;
					break;

				case ORDER_ZXY:
					rotationMatrix[0] = Cy*Cz-Sx*Sy*Sz;
					rotationMatrix[1] = -Cx*Sz;
					rotationMatrix[2] = Cz*Sy+Cy*Sx*Sz;
					rotationMatrix[3] = Cz*Sx*Sy+Cy*Sz;
					rotationMatrix[4] = Cx*Cz;
					rotationMatrix[5] = -Cy*Cz*Sx+Sy*Sz;
					rotationMatrix[6] = -Cx*Sy;
					rotationMatrix[7] = Sx;
					rotationMatrix[8] = Cx*Cy;
					break;

				case ORDER_ZYX:
					rotationMatrix[0] = Cy*Cz;
					rotationMatrix[1] = Cz*Sx*Sy-Cx*Sz;
					rotationMatrix[2] = Cx*Cz*Sy+Sx*Sz;
					rotationMatrix[3] = Cy*Sz;
					rotationMatrix[4] = Cx*Cz+Sx*Sy*Sz;
					rotationMatrix[5] = -Cz*Sx+Cx*Sy*Sz;
					rotationMatrix[6] = -Sy;
					rotationMatrix[7] = Cy*Sx;
					rotationMatrix[8] = Cx*Cy;
					break;

				case ORDER_YXZ:
					rotationMatrix[0] = Cy*Cz+Sx*Sy*Sz;
					rotationMatrix[1] = Cz*Sx*Sy-Cy*Sz;
					rotationMatrix[2] = Cx*Sy;
					rotationMatrix[3] = Cx*Sz;
					rotationMatrix[4] = Cx*Cz;
					rotationMatrix[5] = -Sx;
					rotationMatrix[6] = -Cz*Sy+Cy*Sx*Sz;
					rotationMatrix[7] = Cy*Cz*Sx+Sy*Sz;
					rotationMatrix[8] = Cx*Cy;
					break;

				case ORDER_XZY:
					rotationMatrix[0] = Cy*Cz;
					rotationMatrix[1] = -Sz;
					rotationMatrix[2] = Cz*Sy;
					rotationMatrix[3] = Sx*Sy+Cx*Cy*Sz;
					rotationMatrix[4] = Cx*Cz;
					rotationMatrix[5] = -Cy*Sx+Cx*Sy*Sz;
					rotationMatrix[6] = -Cx*Sy+Cy*Sx*Sz;
					rotationMatrix[7] = Cz*Sx;
					rotationMatrix[8] = Cx*Cy+Sx*Sy*Sz;
					break;

			}

			break;
		}
		
		case EULER_ORIENTATION:
		{
			const float phi = orientationData[0];
			const float theta = orientationData[1];
			const float psi = orientationData[2];

			const float Sx = sinf(phi);
			const float Sy = sinf(theta);
			const float Sz = sinf(psi);
			const float Cx = cosf(phi);
			const float Cy = cosf(theta);
			const float Cz = cosf(psi);
		
			switch(axeOrder.euler) 
			{

				case ORDER_XZX:
					rotationMatrix[0] = Cy;
					rotationMatrix[1] = -Cz*Sy;
					rotationMatrix[2] = Sy*Sz;
					rotationMatrix[3] = Cx*Sy;
					rotationMatrix[4] = Cx*Cy*Cz - Sx*Sz;
					rotationMatrix[5] = -Cz*Sx - Cx*Cy*Sz;
					rotationMatrix[6] = Sx*Sy;
					rotationMatrix[7] = Cx*Sz + Cy*Cz*Sx;
					rotationMatrix[8] = Cx*Cz - Cy*Sx*Sz;
					break;

				case ORDER_XYX:
					rotationMatrix[0] = Cy;
					rotationMatrix[1] = Sy*Sz;
					rotationMatrix[2] = Cz*Sy;
					rotationMatrix[3] = Sx*Sy;
					rotationMatrix[4] = Cx*Cz - Cy*Sx*Sz;
					rotationMatrix[5] = -Cx*Sz - Cy*Cz*Sx;
					rotationMatrix[6] = -Cx*Sy;
					rotationMatrix[7] = Cz*Sx + Cx*Cy*Sz;
					rotationMatrix[8] = Cx*Cy*Cz - Sx*Sy;
					break;

				case ORDER_YXY:
					rotationMatrix[0] = Cx*Cz - Cy*Sx*Sz;
					rotationMatrix[1] = Sx*Sy;
					rotationMatrix[2] = Cx*Sz + Cy*Cz*Sx;
					rotationMatrix[3] = Sy*Sz;
					rotationMatrix[4] = Cy;
					rotationMatrix[5] = -Cz*Sy;
					rotationMatrix[6] = -Cz*Sx - Cx*Cy*Sz;
					rotationMatrix[7] = Cx*Sy;
					rotationMatrix[8] = Cx*Cy*Cz - Sx*Sz;
					break;

				case ORDER_YZY:
					rotationMatrix[0] = Cx*Cy*Cz - Sx*Sz;
					rotationMatrix[1] = -Cx*Sy;
					rotationMatrix[2] = Cz*Sx + Cx*Cy*Sz;
					rotationMatrix[3] = Cz*Sy; 
					rotationMatrix[4] = Cy;
					rotationMatrix[5] = Sy*Sz;
					rotationMatrix[6] = -Cx*Sz - Cy*Cz*Sx;
					rotationMatrix[7] = Sx*Sy;
					rotationMatrix[8] = Cx*Cz - Cy*Sx*Sz;
					break;

				case ORDER_ZYZ:
					rotationMatrix[0] = Cx*Cy*Cz - Sx*Sz; 
					rotationMatrix[1] = -Cz*Sx - Cx*Cy*Sz; 
					rotationMatrix[2] = Cx*Sy;
					rotationMatrix[3] = Cx*Sz + Cy*Cz*Sx;
					rotationMatrix[4] = Cx*Cz - Cy*Sx*Sz; 
					rotationMatrix[5] = Sx*Sy;
					rotationMatrix[6] = -Cz*Sy;
					rotationMatrix[7] = Sy*Sz;
					rotationMatrix[8] = Cy;
					break;

				case ORDER_ZXZ:
					rotationMatrix[0] = Cx*Cz - Cy*Sx*Sz;
					rotationMatrix[1] = -Cx*Sz - Cy*Cz*Sx;
					rotationMatrix[2] = Sx*Sy;
					rotationMatrix[3] = Cz*Sx + Cx*Cy*Sz;
					rotationMatrix[4] = Cx*Cy*Cz - Sx*Sz;
					rotationMatrix[5] = -Cx*Sy;
					rotationMatrix[6] = Sy*Sz;
					rotationMatrix[7] = Cz*Sy;
					rotationMatrix[8] = Cy;
					break;
			}
			
			break;
		}	
		
	}
	

	return rotationMatrix;
}
