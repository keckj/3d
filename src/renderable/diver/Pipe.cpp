#include "Pipe.h"
#include "Dimensions.h"

#include <iostream>

Pipe::Pipe (std::vector<Vec> points) : DynamicSystem(), points(points), cs(points), pas(0.5) {
    groundPosition = Vec(0.0, 0.0, -(HEIGHT_TRUNK + HEIGHT_THIGH));
    springStiffness = 50.0;
    createSystemScene();
}

void Pipe::createSystemScene () {
    // beginning particle is fixed
    Particle *begin = new Particle(points[0], Vec(), 0.0, particleRadius, Vec(0.0, 1.0, 0.0));
    fixe.push_back(begin);

    Particle *pcour, *pprev = begin;
    for (unsigned p = 0; p < points.size() - 1; p++) {
        for (float t = 0; t <= 1; t += pas) {
            Vec cour = cs(p, pas);
            pcour = new Particle(cour, Vec(), particleMass, particleRadius);
            if (p == points.size() - 2) {
                fixe.push_back(pcour);
            } else {
                particles.push_back(pcour);
            }

            springs.push_back(new Spring(pprev, pcour, springStiffness, springInitLength, springDamping));
            pprev = pcour;
        }
    }
}

