#include <GL/glew.h>
#include <QGLWidget>

#include "cubeMap.h"
#include "log.h"
#include "globals.h"
#include <sstream>

CubeMap::CubeMap(std::string const &folder, std::string const &fileNames, std::string const &format) :
    Texture(GL_TEXTURE_CUBE_MAP)
{
	stringstream ss(fileNames);
	std::string fileName;
	for (int i = 0; i < 6; i++) {
		ss >> fileName;
		image[i] = QGLWidget::convertToGLFormat(QImage((folder + fileName).c_str(), format.c_str()));
        
		if(!image[i].bits()) {
            log_console.errorStream() << logTextureHead << "Error while loading image '" << folder + fileName << "' !";
            exit(1);
        }
	}

    log_console.infoStream() << logTextureHead << "Created Cube Map TEXTURE from folder '" << folder << "' with files '" << fileNames << "' .";
}

CubeMap::~CubeMap() {
}

void CubeMap::bindAndApplyParameters(unsigned int location) {

    if(location >= (unsigned int)Globals::glMaxCombinedTextureImageUnits) {
        log_console.errorStream() << logTextureHead << "Trying to bind invalid texture location "
            << location << " (MAX = " << Globals::glMaxCombinedTextureImageUnits << ") !";
        exit(1);
    }

    // Liste des faces successives pour la création des textures de CubeMap
    GLenum cube_map_target[6] = {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
	};

    glActiveTexture(GL_TEXTURE0 + location);
    
	// Configuration de la texture
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureId);

    for (int i = 0; i < 6; i++) {
        if (image[i].bits()) {
            glTexImage2D(cube_map_target[i], 0, GL_RGBA, image[i].width(), image[i].height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image[i].bits());
        }
    }

    log_console.infoStream() << logTextureHead << "Bind Cube Map TEXTURE [id="
        << textureId << "] to texture location " << location << ".";

    log_console.infoStream() << logTextureHead << "Applying " << params.size() << " parameters !";

    applyParameters();

    lastKnownLocation = location;
    textureLocations[location] = textureId;
    locationsHitMap[location]++;
}

