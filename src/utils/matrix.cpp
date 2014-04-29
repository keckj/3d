
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

    float *transpose(const float *M, const unsigned int dim) {
        float *tr = new float[dim*dim];

        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int j = 0; j < dim; j++) {
                tr[i*dim+j] = M[j*dim+i];
            }
        }

        return tr;
    }
    
    float *inverseMat3f(const float *M) {
        float *inv = new float[9];
        float det = M[0] * (M[4] * M[8] - M[7] * M[5]) -
                    M[1] * (M[3] * M[8] - M[5] * M[6]) +
                    M[2] * (M[3] * M[7] - M[4] * M[6]);

        if (fabs(det) < 0.000000001) {
            std::cout << "[Matrix] inverseMat3f: det = 0 !" << std::endl;
            exit(1);
        }

        float invdet = 1 / det;

        inv[0] = (M[4] * M[8] - M[7] * M[5]) * invdet;
        inv[1] = (M[2] * M[7] - M[1] * M[8]) * invdet;
        inv[2] = (M[1] * M[5] - M[2] * M[4]) * invdet;
        inv[3] = (M[5] * M[6] - M[3] * M[8]) * invdet;
        inv[4] = (M[0] * M[8] - M[2] * M[6]) * invdet;
        inv[5] = (M[3] * M[2] - M[0] * M[5]) * invdet;
        inv[6] = (M[3] * M[7] - M[6] * M[4]) * invdet;
        inv[7] = (M[6] * M[1] - M[0] * M[7]) * invdet;
        inv[8] = (M[0] * M[4] - M[3] * M[1]) * invdet;

        return inv;
    }

    float *inverseMat4f(const float *M) {
        float *inv = new float[16];

        float s0 = M[0] * M[5] - M[4] * M[1];
        float s1 = M[0] * M[6] - M[4] * M[2];
        float s2 = M[0] * M[7] - M[4] * M[3];
        float s3 = M[1] * M[6] - M[5] * M[2];
        float s4 = M[1] * M[7] - M[5] * M[3];
        float s5 = M[2] * M[7] - M[6] * M[3];
        float c5 = M[10] * M[15] - M[14] * M[11];
        float c4 = M[9] * M[15] - M[13] * M[11];
        float c3 = M[9] * M[14] - M[13] * M[10];
        float c2 = M[8] * M[15] - M[12] * M[11];
        float c1 = M[8] * M[13] - M[12] * M[10];
        float c0 = M[8] * M[12] - M[12] * M[9];

        float det = (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
        if (fabs(det) < 0.000000001) {
            std::cout << "[Matrix] inverseMat4f: det = 0 !" << std::endl;
            exit(1);
        }
        float invdet = 1.0 / det;

        inv[0] = ( M[5] * c5 - M[6] * c4 + M[7] * c3) * invdet;
        inv[1] = (-M[1] * c5 + M[2] * c4 - M[3] * c3) * invdet;
        inv[2] = ( M[13] * s5 - M[14] * s4 + M[15] * s3) * invdet;
        inv[3] = (-M[9] * s5 + M[10] * s4 - M[11] * s3) * invdet;

        inv[4] = (-M[4] * c5 + M[6] * c2 - M[7] * c1) * invdet;
        inv[5] = ( M[0] * c5 - M[2] * c2 + M[3] * c1) * invdet;
        inv[6] = (-M[12] * s5 + M[14] * s2 - M[15] * s1) * invdet;
        inv[7] = ( M[8] * s5 - M[10] * s2 + M[11] * s1) * invdet;

        inv[8] = ( M[4] * c4 - M[5] * c2 + M[7] * c0) * invdet;
        inv[9] = (-M[0] * c4 + M[1] * c2 - M[3] * c0) * invdet;
        inv[10] = ( M[12] * s4 - M[13] * s2 + M[15] * s0) * invdet;
        inv[11] = (-M[8] * s4 + M[9] * s2 - M[11] * s0) * invdet;

        inv[12] = (-M[4] * c3 + M[5] * c1 - M[6] * c0) * invdet;
        inv[13] = ( M[0] * c3 - M[1] * c1 + M[2] * c0) * invdet;
        inv[14] = (-M[12] * s3 + M[13] * s1 - M[14] * s0) * invdet;
        inv[15] = ( M[8] * s3 - M[9] * s1 + M[10] * s0) * invdet;

        return inv;
    }

    float *mat3f(const float *M) {
        float *res = new float[9];
        res[0] = M[0];
        res[1] = M[1];
        res[2] = M[2];
        res[3] = M[4];
        res[4] = M[5];
        res[5] = M[6];
        res[6] = M[8];
        res[7] = M[9];
        res[8] = M[10];
        return res;
    }

    float *getColumn(const float *M, const unsigned int dim, const unsigned int col) {
        float *res = new float[dim];
        for (unsigned int i = 0; i < dim; i++) {
            res[i] = M[i*dim + col];
        }

        return res;
    }
} 
