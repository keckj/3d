#pragma once

#define NO_AXE 0x00
#define X_AXE 0x01
#define Y_AXE 0x02
#define Z_AXE 0x04
#define RX_AXE 0x08
#define RY_AXE 0x10
#define RZ_AXE 0x20

#define ALL_TRANSLATION 0x07
#define ALL_ROTATION 0x38
#define ALL_AXES 0x3f

#include "types.hpp"
#include "rotation.hpp"

using namespace std;


struct DegreesOfFreedom {
	unsigned int TX : 1;
	unsigned int TY : 1;
	unsigned int TZ : 1;
	unsigned int RX : 1;
	unsigned int RY : 1;
	unsigned int RZ : 1;
};


class Camera {

	public:
		Camera(float near, float far, float fov, float aspectRatio, vec3 &initialPosition, Rotation initialRotation, unsigned int degreesOfFreedom);
		~Camera();

		void lookAt(float x, float y, float z);
		void lookAt(vec3 &vector);

		void rotate(float phi, float theta, float psi);
		void rotate(vec3 &eulerAngles);

		void orientate(float phi, float theta, float psi);
		void orientate(vec3 &eulerAngles);

		void translate(float x, float y, float z);
		void translate(vec3 &vector);

		void moveTo(float x, float y, float z);
		void moveTo(vec3 &vector);

		void setAspectRatio(float aspectRatio);

		vec3 getForwardDirection();
		vec3 getUpwardDirection();
		vec3 getRightDirection();

		mat4 getViewMatrix();
		mat4 getProjectionMatrix();

	private:
		vec3 position;
		Rotation rotation;

		float fov;
		float far, near;
		float aspectRatio;

		DegreesOfFreedom dof;

		mat4 projectionMatrix;
		mat4 viewMatrix;

		void updateProjectionMatrix();
		void updateViewMatrix();
};
