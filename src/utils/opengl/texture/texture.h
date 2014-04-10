
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
		void addParameters(const std::list<Parameter> &paramList);

		const std::list<Parameter> getParameters() const;

		void bindAndApplyParameters(unsigned int location);
		
		unsigned int getTextureId() const;
		int getLastKnownLocation() const;

		bool isBinded() const; //check wether the texture is still linked to its last known location or not

		void generateMipMap();
	
		static void init();
		static std::vector<unsigned int> requestTextures(unsigned int nbRequested);
		static void sortHitMap();
		static void reportHitMap();

	private:
		unsigned int textureId;
		int lastKnownLocation;

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
		static std::vector<std::pair<long, unsigned int> > reversedHitMap; 

		static bool compareFunc(std::pair<long, unsigned int> a, std::pair<long, unsigned int> b);
};

#endif /* end of include guard: TEXTURE_H */
