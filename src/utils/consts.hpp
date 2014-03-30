#ifndef CONSTS_H
#define CONSTS_H

namespace consts {
		
	static const float pi = 3.14159265359f;

	static const float oneDegInRad = 0.0174532925;
	
	static const float identity3[] = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f
	};

	static const float identity4[] = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};

}

#endif /* end of include guard: CONSTS_H */
