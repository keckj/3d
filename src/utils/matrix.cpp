
#include "matrix.h"

namespace Matrix {

	
	float* multMat4f(const float *m1, const float* m2) {
		float *m = new float[16]();

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int k = 0; k < 4; k++) {
					m[4*i+j] += m1[4*i+k] * m2[4*k+j];
				}
			}
		}

		return m;
	}

	
	
	void scaleMat4f(float* M, float alpha) {
		M[0]*=alpha;
		M[5]*=alpha;
		M[10]*=alpha;
	}

	void scaleMat4f(float* M, float alpha, float beta, float gamma) {
		M[0]*=alpha;
		M[5]*=beta;
		M[10]*=gamma;
	}

	void scaleMat4f(float* M, qglviewer::Vec &v) {
		M[0]*=v.x;
		M[5]*=v.y;
		M[10]*=v.z;
	}

	void translateMat4f(float* M, float x, float y, float z) {
		M[3]+=x;
		M[7]+=y;
		M[11]+=z;
	}
	void translateMat4f(float* M, qglviewer::Vec &v) {
		M[3]+=v.x;
		M[7]+=v.y;
		M[11]+=v.z;
	}

	void rotateMat4f(float* M, qglviewer::Quaternion const &quat) {

		double *rot = new double[16];
		float *rotf = new float[16];

		quat.getMatrix(rot);

		for (int i = 0; i < 16; i++) {
			rotf[i] = rot[i];
		}

		const float *res = multMat4f(rotf, M);

		for (int i = 0; i < 16; i++) {
			M[i] = res[i];
		}

		delete [] res;
		delete [] rot;
		delete [] rotf;
	}

}
