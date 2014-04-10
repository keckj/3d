
#include "parameter.h"

Parameter::Parameter(GLenum paramName, int param) :
	_type(ParamType::I), _paramName(paramName)
{
	_params.i = param;
}

Parameter::Parameter(GLenum paramName, float param) :
	_type(ParamType::F), _paramName(paramName)
{
	_params.f = param;
}

Parameter::Parameter(GLenum paramName, int *params) :
	_type(ParamType::IV), _paramName(paramName)
{
	_params.iv = params;
}

Parameter::Parameter(GLenum paramName, float *params) :
	_type(ParamType::FV), _paramName(paramName)
{
	_params.fv = params;
}
		
ParamType Parameter::type() const {
	return _type;
}

GLenum Parameter::paramName() const {
	return _paramName;
}

ParamData Parameter::params() const {
	return _params;
}
