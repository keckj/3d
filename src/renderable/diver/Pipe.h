#ifndef __PIPE_H__
#define __PIPE_H__

#include "dynamicSystem.h"
#include "CardinalSpline.h"

class Pipe : public DynamicSystem {
    public:
        Pipe(std::vector<Vec> points);

    private:
        void createSystemScene ();

        std::vector<Vec> points;
        CardinalSpline cs;
        float pas;
};

#endif

