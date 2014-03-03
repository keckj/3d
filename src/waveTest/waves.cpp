#include <cmath>
#include <iostream> 
#include <cstdio>
#include <ctime>
#include "../viewer.h"
#include "waves.h"


static const int N_MOBILES_X = 256;
static const int N_MOBILES_Z = 256;
static const int N_EXCITERS_X = 10;
static const int N_EXCITERS_Z = 10;
static const float EXCITERS_TIME_LIMIT = 0.10f;


Waves::~Waves() {
    delete[] mobiles;
    delete[] indices;
}

Waves::Waves(float xWidth, float zWidth, float meanHeight, Viewer *v) {

    if (xWidth <= 0.0f || zWidth <= 0.0f || v == NULL) std::cout << "You're doing something stupid!" << std::endl;

    // Indices used for drawing
    nIndices = 6*(N_MOBILES_X-1)*(N_MOBILES_Z-1);
    indices = new GLuint[nIndices];
    int currentIndex = 0;

    this->meanHeight = meanHeight;

    nMobiles = N_MOBILES_X * N_MOBILES_Z;
    mobiles = new Mobile[nMobiles];
    for (int x = 0; x < N_MOBILES_X; x++) {
        for (int z = 0; z < N_MOBILES_Z; z++) {
            int idx = x*N_MOBILES_Z + z;
            mobiles[idx].x = xWidth/(N_MOBILES_X-1) * x;  
            mobiles[idx].y = meanHeight + (rand()%(nMobiles/10) == 0 ? -0.05f : 0.0f); 
            mobiles[idx].z = zWidth/(N_MOBILES_Z-1) * z; 
            // init with flat sea => all normals along y axis
            mobiles[idx].nx = 0.0f;
            mobiles[idx].ny = 1.0f;
            mobiles[idx].nz = 0.0f;
            // TODO: find a good formula 
            //if (rand()%1000 == 0) {    
            //if (x == 0 && z == 0) {
            if (false) { // no exciters, use 'raindrops' to kickstart (see above)
                mobiles[idx].isExciter = true;
                mobiles[idx].frequency = 50.0f; // Hz
                mobiles[idx].amplitude = 0.1f * xWidth;
            } else {
                mobiles[idx].isExciter = false;
            }
            mobiles[idx].speed = 0.0f;
            mobiles[idx].nextY = 0.0f;

            if (x < N_MOBILES_X-1 && z < N_MOBILES_Z-1) { 
                // 1st triangle
                indices[currentIndex++] = idx;                  // up left
                indices[currentIndex++] = idx+1;                // up right
                indices[currentIndex++] = idx+1+N_MOBILES_Z;    // down right
                // 2nd triangle
                indices[currentIndex++] = idx;                  // up left
                indices[currentIndex++] = idx+1+N_MOBILES_Z;    // down right
                indices[currentIndex++] = idx+N_MOBILES_Z;      // down left
            }
        }
    }

    viewer = v;
    time = 0.0f;
}


void Waves::draw() {

    //Enable the vertex array functionality
    //(we don't use VBOs because we modify the arrays each frame when animating)
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(	3,
            GL_FLOAT,
            sizeof(Mobile),
            mobiles);
    glNormalPointer(	GL_FLOAT,
            sizeof(Mobile),
            &mobiles[0].nx);

    glDrawElements(GL_TRIANGLES, nIndices, GL_UNSIGNED_INT, indices);
}


void Waves::animate() {

    // Approximate time passed since last frame was drawn (at the start we have 0 FPS so we need to cheat a bit)
    //float deltaT = 1.0 / (viewer->currentFPS() > 0 ? viewer->currentFPS() : 50);
    float deltaT = 1.0 / 50;
    time += deltaT;

    // Coordinates
    for (int x = 0; x < N_MOBILES_X; x++) {
        for (int z = 0; z < N_MOBILES_Z; z++) {
            int idx = x*N_MOBILES_Z + z;
            if (mobiles[idx].isExciter && time < EXCITERS_TIME_LIMIT) {
                /*mobiles[idx].nextY = mobiles[idx].amplitude/3 * sin(time * mobiles[idx].frequency) + 
                                     mobiles[idx].amplitude/6 * sin(time * mobiles[idx].frequency*2) +
                                     mobiles[idx].amplitude * sin(time * mobiles[idx].frequency/2) +
                                     meanHeight;*/
                mobiles[idx].nextY = mobiles[idx].amplitude * sin(time * mobiles[idx].frequency) + meanHeight; 
            } else {
                float xm, xp, zm, zp;
                int cnt = 4;
                if (x == 0) {
                    xm = 0.0f;
                    cnt--; 
                } else {
                    xm = mobiles[idx-N_MOBILES_Z].y;
                }
                if (x == N_MOBILES_X-1) {
                    xp =  0.0f;
                    cnt--;
                } else {
                    xp = mobiles[idx+N_MOBILES_Z].y;
                }
                if (z == 0) {
                    zm = 0.0f;
                    cnt--;
                } else {
                    zm = mobiles[idx-1].y;
                }
                if (z == N_MOBILES_Z-1) {
                    zp = 0.0f;
                    cnt--;
                } else {
                   zp = mobiles[idx+1].y;
                }
                float deltaY = xm + xp + zm + zp - cnt*mobiles[idx].y;
                mobiles[idx].speed += deltaY * deltaT / 0.001;
                mobiles[idx].nextY += mobiles[idx].speed * deltaT;
            }
        }
    }
    // Update y coordinate + normals
    // --------------- NOTE: We might use a Vector class in the future with everything implemeted ----------
    for (int x = 0; x < N_MOBILES_X; x++) {
        for (int z = 0; z < N_MOBILES_Z; z++) {
            int idx = x*N_MOBILES_Z + z;
            // Update y
            mobiles[idx].y = mobiles[idx].nextY;

            // Approximation of vectors in the wave surface
            float v1[3], v2[3];
            // x direction
            Mobile mp = (x >= N_MOBILES_X-1 ? mobiles[idx] : mobiles[idx+N_MOBILES_Z]);
            Mobile mm = (x == 0             ? mobiles[idx] : mobiles[idx-N_MOBILES_Z]);
            v1[0] = mp.x - mm.x;
            v1[1] = mp.y - mm.y;
            v1[2] = mp.z - mm.z;
            // z direction
            mp =        (z >= N_MOBILES_Z-1 ? mobiles[idx] : mobiles[idx+1]);
            mm =        (z == 0             ? mobiles[idx] : mobiles[idx-1]);
            v2[0] = mp.x - mm.x;
            v2[1] = mp.y - mm.y;
            v2[2] = mp.z - mm.z;
            // Normal = cross_product(v1,v2)
            float n[3];
            n[0] = v1[1]*v2[2] - v1[2]*v2[1];
            n[1] = v1[2]*v2[0] - v1[0]*v2[2];
            n[2] = v1[0]*v2[1] - v1[1]*v2[0];
            // Normalize the resulting vector
            float norm = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            if (norm == 0.0f) {
                mobiles[idx].nx = 0.0f;
                mobiles[idx].ny = 0.0f;
                mobiles[idx].nz = 0.0f;
            } else {
                mobiles[idx].nx = n[0]/norm;
                mobiles[idx].ny = (viewer->camera()->position()[1] > 0 ? -n[1]/norm : n[1]/norm);
                mobiles[idx].nz = n[2]/norm;
            }
        }
    }

    //std::cout << "fps=" << viewer->currentFPS() << std::endl;
    //std::cout << "time=" << time << std::endl;
}

