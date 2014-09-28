#ifndef CARDINALSPLINE_H
#define CARDINALSPLINE_H

#include <vector>

#include <QGLViewer/vec.h>
using namespace qglviewer;

class CardinalSpline {
    public:
        CardinalSpline (std::vector<Vec> points, float k = 0.5);

        Vec operator() (unsigned int k, float t);
        Vec operator() (float t);

    private:
        std::vector<Vec> points;
        std::vector<Vec> tangents;
        float k;

        float h00 (float t);
        float h10 (float t);
        float h01 (float t);
        float h11 (float t);
};

#endif

