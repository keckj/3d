
#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <QGLViewer/vec.h>
#include <QGLViewer/quaternion.h>

namespace Matrix {
		
		float *multMat4f(const float *m1, const float* m2);
	
		void scaleMat4f(float *M, float alpha);
		void scaleMat4f(float *M, float alpha, float beta, float gamma);
		void scaleMat4f(float *M, qglviewer::Vec &v);
		
		void translateMat4f(float *M, float x, float y, float z);
		void translateMat4f(float *M, qglviewer::Vec &v);

		void setOffsetMat4f(float *M, float x, float y, float z);
		void setOffsetMat4f(float *M, qglviewer::Vec &v);

		void setRotationMat4f(float *M, qglviewer::Quaternion rot, float scale = 1.0f);

		void rotateMat4f(float *M, qglviewer::Quaternion const &quat);

        float *transpose(const float *M, const unsigned int dim = 4);

        float *inverseMat3f(const float *M);

        float *inverseMat4f(const float *M);

        // Equivalent to mat3(mat4) in glsl
        float *mat3f(const float *M);

        // Row major matrix
        float *getColumn(const float *M, const unsigned int dim, const unsigned int col);
}

#endif /* end of include guard: __MATRIX_H__ */

