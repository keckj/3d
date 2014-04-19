#ifndef PARAMETER_H
#define PARAMETER_H

#include <GL/glut.h>

enum ParamType {F,I,IV,FV};

union ParamData {
	int i;
	float f;
	int *iv;
	float *fv;
};

class Parameter {
	public:
		explicit Parameter(GLenum paramName, int param);
		explicit Parameter(GLenum paramName, float param);

		explicit Parameter(GLenum paramName, int *params);
		explicit Parameter(GLenum paramName, float *params);

		ParamType type() const;
		GLenum paramName() const;
		ParamData params() const;

	private:
		ParamType _type;
		GLenum _paramName;
		ParamData _params;
};

#endif /* end of include guard: PARAMETER_H */
