#include "CardinalSpline.h"

#include <cassert>

CardinalSpline::CardinalSpline (std::vector<Vec> points, float k) : points(points), k(k) {
    assert(points.size() >= 4);

    // Computation of the n-2 tangents
    for (unsigned int i = 1; i <= points.size() - 2; i++) {
        Vec t = k * (points[i+1] - points[i-1]);
        tangents.push_back(t);
    }
}

Vec CardinalSpline::operator() (unsigned int n, float t) {
    assert(n < points.size());
    assert(t >= 0 && t <= 1);

    return (points[n] * h00(t) + tangents[n] * h10(t) + points[n+1] * h01(t) + tangents[n+1] * h11(t));
}

Vec CardinalSpline::operator() (float t) {
    int n = floor(t);

    return operator()(n, t - n);
}

// Hermite Basis functions
float CardinalSpline::h00 (float t) {
    return ((1 + 2 * t) * (1 - t) * (1 - t));
}

float CardinalSpline::h10 (float t) {
    return (t * (1 - t) * (1 - t));
}

float CardinalSpline::h01 (float t) {
    return (t * t * (3 - 2 * t));
}

float CardinalSpline::h11 (float t) {
    return (t * t * (t - 1));
}

