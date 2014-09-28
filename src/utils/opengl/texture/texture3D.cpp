

#include <GL/glew.h>
#include <QGLWidget>

#include "texture3D.h"
#include "log.h"
#include "globals.h"


Texture3D::Texture3D(unsigned int width, unsigned int height, unsigned int length,
		GLint internalFormat, 
		void *sourceData, GLenum sourceFormat, GLenum sourceType) :
	Texture(GL_TEXTURE_3D), 
	_width(width), _height(height), _length(length),
	_texels(sourceData), _internalFormat(internalFormat),
	_sourceFormat(sourceFormat), _sourceType(sourceType)
{
	log_console.infoStream() << logTextureHead << "Created 3D TEXTURE with size " 
		<< _width << "x" << _height << "x" << _length << " !";
}

Texture3D::~Texture3D() {
}

void Texture3D::bindAndApplyParameters(unsigned int location) {

	if(location >= (unsigned int)Globals::glMaxCombinedTextureImageUnits) {
		log_console.errorStream() << logTextureHead << "Trying to bind invalid texture location " 
			<< location << " (MAX = " << Globals::glMaxCombinedTextureImageUnits << ") !";
		exit(1);
	}

	glActiveTexture(GL_TEXTURE0 + location);
	glBindTexture(textureType, textureId);

	log_console.infoStream() << logTextureHead << "Bind 3D TEXTURE [id=" 
		<< textureId << "] to texture location " << location << ".";

	glTexImage3D(GL_TEXTURE_3D, 0, _internalFormat, _width, _height, _length, 0,
			_sourceFormat, _sourceType, _texels);

	log_console.infoStream() << logTextureHead << "Updated texture data !"; 

	log_console.infoStream() << logTextureHead << "Applying " << params.size() << " parameters !";

	applyParameters();

	if(mipmap) {
		glGenerateMipmap(textureType);
		log_console.infoStream() << logTextureHead << "Generating mipmap !";
	}

	lastKnownLocation = location;
	textureLocations[location] = textureId;
	locationsHitMap[location]++;
}

void Texture3D::setData(void *data, GLenum sourceFormat, GLenum sourceType) {
	_texels = data;
	_sourceFormat = sourceFormat;
	_sourceType = sourceType;
}
