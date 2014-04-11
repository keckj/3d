
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

		void rotateMat4f(float *M, qglviewer::Quaternion const &quat);
}

#endif /* end of include guard: __MATRIX_H__ */

