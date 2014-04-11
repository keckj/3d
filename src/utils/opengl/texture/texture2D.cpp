
#include <QGLWidget>

#include "texture2D.h"
#include "log.h"
#include "globals.h"

Texture2D::Texture2D(std::string const &src, std::string const &type) :
	Texture(GL_TEXTURE_2D), src(src), type(type)
{
	image = QGLWidget::convertToGLFormat(QImage(src.c_str(),type.c_str()));

	if(!image.bits()) {
		log_console.errorStream() << logTextureHead << "Error while loading image '" << src << "' !";
		exit(1);
	}

	log_console.infoStream() << logTextureHead << "Created 2D TEXTURE from '" << src << "' with type '" << type << "'.";
}

Texture2D::~Texture2D() {
}

void Texture2D::bindAndApplyParameters(unsigned int location) {

	if(location >= (unsigned int)Globals::glMaxCombinedTextureImageUnits) {
		log_console.errorStream() << logTextureHead << "Trying to bind invalid texture location " 
			<< location << " (MAX = " << Globals::glMaxCombinedTextureImageUnits << ") !";
		exit(1);
	}

	glActiveTexture(GL_TEXTURE0 + location);
	glBindTexture(textureType, textureId);

	log_console.infoStream() << logTextureHead << "Bind 2D TEXTURE [id=" 
		<< textureId << "] to texture location " << location << ".";

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0,
			GL_RGBA, GL_UNSIGNED_BYTE, image.bits());

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

