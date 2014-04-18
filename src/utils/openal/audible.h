
#ifndef AUDIBLE_H
#define AUDIBLE_H

#include "headers.h"

class Audible {
	
	public:
		Audible(std::string const &sourcePath, 
				const qglviewer::Vec &initialSourcePosition,
				float pitch = 1.0f, float gain = 1.0f,
				bool loop = true);

		~Audible();
		
		void setPitch(float pitch);
		void setGain(float gain);
		void setLoop(bool loop);
	
		void setSourcePosition(const qglviewer::Vec &pos);
		void setSourceVelocity(const qglviewer::Vec &vel);
		void setSourceOrientation(const qglviewer::Vec &v1, const qglviewer::Vec &v2);

		void playSource();
		void pauseSource();
		void stopSource();
		void rewindSource();
		
		static void setListenerPosition(const qglviewer::Vec &pos);
		static void setListenerVelocity(const qglviewer::Vec &vel);
		static void setListenerOrientation(const qglviewer::Vec &v1, const qglviewer::Vec &v2);
		
		static void initOpenALContext();

	private:
		std::string _sourcePath;
		unsigned int _source;
		unsigned int _buffer;
		
		static bool _init;
};


#endif /* end of include guard: AUDIBLE_H */
