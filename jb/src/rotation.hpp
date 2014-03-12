#pragma once

#include <string>

enum RotationType {
	INTRINSIC_ROTATION = 0,
	EXTRINSIC_ROTATION
};

enum OrientationType {
	EULER_ORIENTATION = 0,
	TAIT_BRYAN_ORIENTATION,
	QUATERNION_ORIENTATION
};

enum EulerOrder {
	ORDER_XZX = 0,
	ORDER_XYX,
	ORDER_YXY,
	ORDER_YZY,
	ORDER_ZYZ,
	ORDER_ZXZ
};

enum TaitBryanOrder {
	ORDER_XYZ = 0,
	ORDER_YZX,
	ORDER_ZXY,
	ORDER_ZYX,
	ORDER_YXZ,
	ORDER_XZY
};

union Order {
	EulerOrder euler;
	TaitBryanOrder tb;
};

class Rotation {
	public:
		Rotation();
		~Rotation();

		Rotation(const Rotation& other);
		Rotation& operator= (const Rotation& other);

		Rotation(RotationType rotationType, OrientationType orientationType, EulerOrder eulerOrder, float *eulerAngles);
		Rotation(RotationType rotationType, OrientationType orientationType, TaitBryanOrder taitBryanOrder, float *cardanAngles);

		RotationType rotationType;
		OrientationType orientationType;
		Order axeOrder;

		float *orientationData;

		float *getRotationMatrix();
		
		std::string toString();


	private:
		void initializeRotation(RotationType rotationType, OrientationType orientationType, Order axeOrder, float *orientationData);
};


const char* toStringRotationType(RotationType t);
const char* toStringOrientationType(OrientationType t);
const char* toStringEulerOrder(EulerOrder o);
const char* toStringTaitBryanOrder(TaitBryanOrder o);
const char* toStringAxeOrder(Order o, OrientationType t);

