#include <GL/glew.h>
#include <QGLWidget>

#include "textureCube.h"
#include "log.h"
#include "globals.h"

#define GL_TEXTURE_CUBE_MAP_ARB             0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB  0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB  0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB  0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB  0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB  0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB  0x851A

TextureCube::TextureCube(std::string const &path) :
    Texture(GL_TEXTURE_CUBE_MAP_ARB), path(path)
{
    // Test de l'extension GL_ARB_texture_cube_map
    char* extensions = (char*) glGetString(GL_EXTENSIONS);

    if(strstr(extensions, "GL_ARB_texture_cube_map") == NULL) {
        exit(1);
    }

    // TODO: use path
    image[0] = QGLWidget::convertToGLFormat(QImage("textures/skybox2/XN.png"));
    image[1] = QGLWidget::convertToGLFormat(QImage("textures/skybox2/XP.png"));
    image[2] = QGLWidget::convertToGLFormat(QImage("textures/skybox2/YN.png"));
    image[3] = QGLWidget::convertToGLFormat(QImage("textures/skybox2/YP.png"));
    image[4] = QGLWidget::convertToGLFormat(QImage("textures/skybox2/ZN.png"));
    image[5] = QGLWidget::convertToGLFormat(QImage("textures/skybox2/ZP.png"));

    for (int i = 0; i < 6; i++) {
        if(!image[i].bits()) {
            log_console.errorStream() << logTextureHead << "Error while loading image '" << path << "' !";
            exit(1);
        }
    }

    log_console.infoStream() << logTextureHead << "Created Cube TEXTURE from '" << path << "'.";
}

TextureCube::~TextureCube() {
}

void TextureCube::bindAndApplyParameters(unsigned int location) {

    if(location >= (unsigned int)Globals::glMaxCombinedTextureImageUnits) {
        log_console.errorStream() << logTextureHead << "Trying to bind invalid texture location "
            << location << " (MAX = " << Globals::glMaxCombinedTextureImageUnits << ") !";
        exit(1);
    }

    // Liste des faces successives pour la création des textures de CubeMap
    GLenum cube_map_target[6] = {
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB,
        GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB
    };

    glActiveTexture(GL_TEXTURE0 + location);
    // Configuration de la texture
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, textureId);

    for (int i = 0; i < 6; i++) {
        if (image[i].bits()) {
            glTexImage2D(cube_map_target[i], 0, GL_RGBA, image[i].width(), image[i].height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image[i].bits());
        }
    }

    // Configuration des parametres de la texture
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP );
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP );

    log_console.infoStream() << logTextureHead << "Bind Cube TEXTURE [id="
        << textureId << "] to texture location " << location << ".";

    log_console.infoStream() << logTextureHead << "Applying " << params.size() << " parameters !";

    applyParameters();

    lastKnownLocation = location;
    textureLocations[location] = textureId;
    locationsHitMap[location]++;
}

