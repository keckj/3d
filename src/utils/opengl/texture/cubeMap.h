#ifndef CUBEMAP_H
#define CUBEMAP_H

#include "texture.h"

class CubeMap : public Texture {

    public:
		//files order : POS_X NEG_X POS_Y NEG_Y POS_Z NEG_Z
		CubeMap(std::string const &folder, std::string const &fileNames, const std::string &format);
        ~CubeMap();

        void bindAndApplyParameters(unsigned int location);

    private:
        QImage image[6];

};

#endif /* end of include guard: CUBEMAP_H */

