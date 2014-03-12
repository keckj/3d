
#include "camera.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <cmath>

using namespace std;

Camera::Camera(float near, float far, float fov, float aspectRatio, vec3 &initialPosition, Rotation initialRotation, unsigned int degreesOfFreedom) {

	Camera::near = near;
	Camera::far = far; 
	Camera::fov = fov;
	Camera::aspectRatio = aspectRatio;

	Camera::position = initialPosition;
	Camera::rotation = initialRotation;

	Camera::dof = {degreesOfFreedom, degreesOfFreedom >> 1, 
		degreesOfFreedom >> 2, degreesOfFreedom >> 3, 
		degreesOfFreedom >> 4, degreesOfFreedom >> 5};
	
	Camera::projectionMatrix = new float[16];
	Camera::viewMatrix = new float[16];

	updateProjectionMatrix();
	updateViewMatrix();

	clog << "\nCamera created at position " 
		<< printVec3(initialPosition)
		<< initialRotation.toString()
		<< "\nDegrees of freedom :"
		<< "\n\tTX=" << dof.TX << "\tTY=" << dof.TY << "\tTZ=" << dof.TZ
		<< "\n\tRX=" << dof.RX << "\tRY=" << dof.RY << "\tRZ=" << dof.RZ
		<< "\n\nView data :"
		<< "\n\tnear=" << near << "\t far=" << far << "\tfov=" << fov << "\taspectRatio=" << aspectRatio;

	clog << "\n\nProjection matrix :\n" <<printMat4(projectionMatrix)  
	     << "\n\nViewmatrix :\n" << printMat4(viewMatrix);

	clog << "\n\nCamera orientation :\n\tFRONT -> " << printVec3(getForwardDirection()) << " \tUP -> " << printVec3(getUpwardDirection()) << "\tRIGHT -> " << printVec3(getRightDirection());

	clog << "\n";
}

Camera::~Camera() {
	delete [] viewMatrix;
	delete [] projectionMatrix;
}
		
void Camera::updateProjectionMatrix() {

	for(int i=0; i < 16; i++)
		projectionMatrix[i] = 0;

	float range = tan(fov * 0.5f) * near;

	projectionMatrix[0] = (2.0f * near) / (range * aspectRatio + range * aspectRatio);
	projectionMatrix[5] = near / range;
	projectionMatrix[10] = -(far + near) / (far - near);
	projectionMatrix[11] = -(2.0f * far * near) / (far - near);
	projectionMatrix[14] = -1;
}

void Camera::updateViewMatrix() {
	
	// The rotation matrix 
	float *rotationMatrix = rotation.getRotationMatrix();

	viewMatrix[0] = rotationMatrix[0];
	viewMatrix[1] = rotationMatrix[1];
	viewMatrix[2] = rotationMatrix[2];

	viewMatrix[4] = rotationMatrix[3];
	viewMatrix[5] = rotationMatrix[4];
	viewMatrix[6] = rotationMatrix[5];

	viewMatrix[8] = rotationMatrix[6];
	viewMatrix[9] = rotationMatrix[7];
	viewMatrix[10] = rotationMatrix[8];

	// The rest of the matrix
	viewMatrix[3] =  - position.x; 
	viewMatrix[7] =  - position.y;
	viewMatrix[11] = - position.z; 

	viewMatrix[12] =   0.0f;
	viewMatrix[13] =   0.0f;
	viewMatrix[14] =   0.0f;
	viewMatrix[15] =   1.0f;
}

void Camera::rotate(float phi, float theta, float psi) {
	rotation.orientationData[0] += phi * dof.RX;
	rotation.orientationData[1] += theta * dof.RY;
	rotation.orientationData[2] += psi * dof.RZ;

	updateViewMatrix();
}

void Camera::rotate(vec3 &eulerAngles) {
	rotate(eulerAngles.x, eulerAngles.y, eulerAngles.z);
}

void Camera::orientate(float phi, float theta, float psi) {
	rotation.orientationData[0] = phi;
	rotation.orientationData[1] = theta;
	rotation.orientationData[2] = psi;
	
	updateViewMatrix();
}

void Camera::orientate(vec3 &eulerAngles) {
	orientate(eulerAngles.x, eulerAngles.y, eulerAngles.z);
}

void Camera::translate(float x, float y, float z) {
	position.x += x * dof.TX;
	position.y += y * dof.TY;
	position.z += z * dof.TZ;
	
	updateViewMatrix();
}
void Camera::translate(vec3 &vector) {
	translate(vector.x, vector.y, vector.z);
}

void Camera::moveTo(float x, float y, float z) {
	position.x = x;
	position.y = y;
	position.z = z;
	
	updateViewMatrix();
}
void Camera::moveTo(vec3 &vector) {
	moveTo(vector.x, vector.y, vector.z);
}



void Camera::setAspectRatio(float aspectRatio) {
	this->aspectRatio = aspectRatio;
	updateProjectionMatrix();
}

vec3 Camera::getForwardDirection() {

	vec3 dir;
	dir.x = -viewMatrix[8];
	dir.y = -viewMatrix[9];
	dir.z = -viewMatrix[10];

	return dir;
}

vec3 Camera::getUpwardDirection() {

	vec3 dir;

	dir.x = viewMatrix[4];
	dir.y = viewMatrix[5];
	dir.z = viewMatrix[6];

	return dir;
}

vec3 Camera::getRightDirection() {

	vec3 dir;

	dir.x = viewMatrix[0];
	dir.y = viewMatrix[1];
	dir.z = viewMatrix[2];

	return dir;
}

mat4 Camera::getViewMatrix() {
	return viewMatrix;
}

mat4 Camera::getProjectionMatrix() {
	return projectionMatrix;
}
