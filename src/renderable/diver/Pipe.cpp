#include "Pipe.h"
#include <iostream>

Pipe::Pipe (std::vector<Vec> points) : DynamicSystem(), points(points), cs(points), pas(1) {
}

void Pipe::createSystemScene () {
    std::cout << "pipe create" << std::endl;

    // beginning is fixed
    Particle *begin = new Particle(points[0], Vec(), 0.0, particleRadius, Vec(0.0, 1.0, 0.0));
    fixed.push_back(begin);

    Particle *pcour, *pprev = begin;
    for (unsigned p = 0; p < points.size() - 1; p++) {
        for (float t = 0; t <= 1; t += pas) {
            Vec cour = cs(p, pas);
            pcour = new Particle(cour, Vec(), particleMass, particleRadius);
            if (p == points.size() - 2) {
                fixed.push_back(pcour);
            } else {
                particles.push_back(pcour);
            }

            springs.push_back(new Spring(pprev, pcour, springStiffness, springInitLength, springDamping));
            pprev = pcour;
        }
    }
}

