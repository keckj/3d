
#include "headers.h"
#include "utils.h"
#include <cmath>

#include <sstream>

namespace Utils {

	const std::string toStringMemory(unsigned long bytes) {
		std::stringstream ss;

		const char prefix[] = {' ', 'K', 'M', 'G', 'T', 'P'};
		unsigned long val = 1;
		for (int i = 0; i < 6; i++) {
			if(bytes < 1024*val) {
				ss << round(100*(float)bytes/val)/100.0 << prefix[i] << 'B';
				break;
			}
			val *= 1024;
		}

		const std::string str(ss.str());
		return str;
	}

	void checkFrameBufferStatus() {

		switch(glCheckFramebufferStatus(GL_FRAMEBUFFER))  {
			case GL_FRAMEBUFFER_COMPLETE:
				log_console.infoStream() << "Framebuffer complete !";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
				log_console.errorStream() << "Framebuffer incomplete layer targets !";
				exit(1);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
				log_console.errorStream() << "Framebuffer incomplete attachement !";
				exit(1);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
				log_console.errorStream() << "Framebuffer missing attachment !";
				exit(1);
				break;
			case GL_FRAMEBUFFER_UNSUPPORTED:
				log_console.errorStream() << "Framebuffer unsupported !";
				exit(1);
				break;
			default:
				log_console.errorStream() << "Something went wrong when configuring the framebuffer !";
				exit(1);
		}
	}
}
