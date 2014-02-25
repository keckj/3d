#include <cmath>
#include <iostream> 
#include <cstdio>
#include "waves.h"

void gen_sin_sum(int n, double max_height, double offset);

/*int main(void) {
    gen_sin_sum(3, 1, 0.0);
    std::cout << std::endl;
    //gen_sin_sum(2, 1, 1.0);
    return 0;
}*/

void gen_sin_sum(int n, double max_height, double offset) {
    
    double tab[256] = {0};
    for (int i = 0; i < 256; i++) {
        for (int k = 1; k <= n; k++) {
            tab[i] += max_height/std::pow(2.f,k) * sin((2*M_PI)/k + (double)i + offset);
        }
    }

    printf("x <- seq(0, 255)\n");
    printf("h <- c(");

    for (int i = 0; i < 256; i++) {
        if (i == 255) {
            printf("%f)\n", tab[i]);
        } else {
            printf("%f, ", tab[i]);
        }
    }

    printf("plot(x,h, type=\"n\")\n");
    printf("lines(x,h)\n");

}



