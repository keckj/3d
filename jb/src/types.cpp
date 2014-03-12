#include "types.hpp"

#include <sstream>

string printMat3(mat3 m) {
	
	stringstream s;
	
	for(int i=0; i < 9; i++) { 
		s << ((i%3 == 0 && i != 0) ? "\n\t" : "\t");
		s << m[i];
	}
	
	return s.str();
}

string printMat4(mat4 m) {
	
	stringstream s;
	
	for(int i=0; i < 16; i++) {
		s << ((i%4 == 0 && i != 0) ? "\n\t" : "\t");
		s << m[i];
	}
	
	return s.str();
}



string printVec3(vec3 v) {

	stringstream s;
	
	s << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	
	return s.str();
}

string printVec4(vec4 v) {

	stringstream s;
	
	s << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.t << ")";
	
	return s.str();
}
