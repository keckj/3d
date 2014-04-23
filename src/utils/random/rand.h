
#include <cstdlib>

class Random {
        public :
                static float randf() {
                        return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                }


                static float randf(float LO, float HI) {
                        return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                }
                
                static int randi(int LO, int HI) {
                        return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
                }
};
