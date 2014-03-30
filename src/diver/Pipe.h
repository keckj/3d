#ifndef PIPE_H
#define PIPE_H

#include "../physics/dynamicSystem.h"
#include "../splines/CardinalSpline.h"

class Pipe : public DynamicSystem {
    public:
        Pipe (std::vector<Vec> points);

    private:
        void createSystemScene ();

        std::vector<Vec> points;
        CardinalSpline cs;
        float pas;
};

#endif

