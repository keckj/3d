#ifndef TEXTURECUBE_H
#define TEXTURECUBE_H

#include "texture.h"

class TextureCube : public Texture {

    public:
        TextureCube(std::string const &path = "textures/skybox2/");
        ~TextureCube();

        void bindAndApplyParameters(unsigned int location);

    private:
        QImage image[6];
        const std::string path;

};

#endif /* end of include guard: TEXTURECUBE_H */

