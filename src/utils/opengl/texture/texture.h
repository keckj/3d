
#ifndef TEXTURE_H
#define TEXTURE_H

#include <list>
#include <string>
#include <map>
#include <QImage>

#include "parameter.h"

class Texture {
	
	public:
		Texture(std::string const &src, std::string const &type, GLenum target); //TEXT1D & TEXT2D
		~Texture();

		void addParameter(Parameter param);
		void addParameters(const std::list<Parameter> &params);

		std::list<Parameter> getParameters() const;

		void bindAndApplyParameters(unsigned int location);
		
		unsigned int getTextureId() const;
		int getTextureLocation() const;

		void generateMipMap();
	
		static void init();
		static std::vector<unsigned int> requestTextures(unsigned int nbRequested);
		static void sortHitMap();

	private:
		unsigned int textureId;

		const std::string src;
		const std::string type;

		GLenum textureType;
		std::list<Parameter> params;
				
		QImage image;
		std::string logTextureHead;

		bool mipmap;

		//texture must be linked
		void applyParameters() const;
	
		static bool _init;
		static std::vector<int> textureLocations;
		static std::map<unsigned int, long> locationsHitMap;
		static std::map<long, unsigned int> reverseOrderedLocationsHitMap;
};

#endif /* end of include guard: TEXTURE_H */
