
#ifndef TEXTURE2D_H
#define TEXTURE2D_H

#include "texture.h"

class Texture2D : public Texture {

	public: 
		Texture2D(std::string const &src, std::string const &type);
		~Texture2D();

		void bindAndApplyParameters(unsigned int location);

	private:
		QImage image;
		const std::string src;
		const std::string type;

};

#endif /* end of include guard: TEXTURE2D_H */
