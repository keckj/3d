
#include "audible.h"
		
bool Audible::_init = false;
ALCdevice* Audible::_devices = 0;
ALCcontext* Audible::_context = 0;

Audible::Audible(std::string const &sourcePath, 
		const qglviewer::Vec &initialSourcePosition,
		float pitch, float gain,
		bool loop) :
	_sourcePath(sourcePath), _source(0), _buffer(0)
{
	
	alGenSources((ALuint)1, &_source);
	
	_buffer = alutCreateBufferFromFile(sourcePath.c_str());
	alSourcei(_source, AL_BUFFER, _buffer);
	
	setSourcePosition(initialSourcePosition);
	setPitch(pitch);
	setGain(gain);
	setLoop(loop);
}

Audible::~Audible() {
	alDeleteSources(1, &_source);
	alDeleteBuffers(1, &_buffer);
}

void Audible::setPitch(float pitch) {
	alSourcef(_source, AL_PITCH, pitch);
}
void Audible::setGain(float gain) {
	alSourcef(_source, AL_GAIN, gain);
}
void Audible::setLoop(bool loop) {
	alSourcei(_source, AL_LOOPING, loop ? AL_TRUE : AL_FALSE);
}

void Audible::setSourcePosition(const qglviewer::Vec &pos) {
	alSource3f(_source, AL_POSITION, pos.x, pos.y, pos.z);
}

void Audible::setSourceVelocity(const qglviewer::Vec &vel) {
	alSource3f(_source, AL_VELOCITY, vel.x, vel.y, vel.z);
}

void Audible::setSourceOrientation(const qglviewer::Vec &v1, const qglviewer::Vec &v2) {
	ALfloat orientation[] = {(float)v1.x, (float)v1.y, (float)v1.z, (float)v2.x, (float)v2.y, (float)v2.z};
	alSourcefv(_source, AL_ORIENTATION, orientation);
}

void Audible::playSource() { alSourcePlay(_source); }
void Audible::pauseSource() { alSourcePause(_source); }
void Audible::stopSource() { alSourceStop(_source); }
void Audible::rewindSource() { alSourceRewind(_source); }

void Audible::setListenerPosition(const qglviewer::Vec &pos) {
	alListener3f(AL_POSITION, pos.x, pos.y, pos.z);
}

void Audible::setListenerVelocity(const qglviewer::Vec &vel) {
	alListener3f(AL_VELOCITY, vel.x, vel.y, vel.z);
}

void Audible::setListenerOrientation(const qglviewer::Vec &v1, const qglviewer::Vec &v2) {
	ALfloat orientation[] = {(float)v1.x, (float)v1.y, (float)v1.z, (float)v2.x, (float)v2.y, (float)v2.z};
	alListenerfv(AL_ORIENTATION, orientation);
}


void Audible::initOpenALContext() {
	log_console.infoStream() << "[OpenAL Init]";
		
	ALboolean enumeration;
	enumeration = alcIsExtensionPresent(NULL, "ALC_ENUMERATION_EXT");
	if (enumeration == AL_FALSE) {
		log_console.infoStream() << "\nEnumerating devices is not supported !";
	}
	else {
		const ALCchar *device = alcGetString(NULL, ALC_DEVICE_SPECIFIER);
		// enumeration supported
		log_console.infoStream() << "\tDevice list :";
		printf("\t\t\t----------\n");
		printf("\t\t\t%s\n", device);
		printf("\t\t\t----------\n");
	}
	
	_devices = alcOpenDevice(NULL);
	if(!_devices) {
		log_console.errorStream() << "\tFailed to open openAL devices !";
		exit(1);
	}
	else {
		log_console.infoStream() << "\tInitialized OpenAL devices !";
	}

	_context = alcCreateContext(_devices, NULL);
	if (!alcMakeContextCurrent(_context)) {
		log_console.errorStream() << "\tFailed to initialize openAL context !";
		exit(1);
	}
	else {
		log_console.infoStream() << "\tInitialized an OpenAL context !";
	}
}
		
void Audible::closeOpenALContext() {
	alcMakeContextCurrent(NULL);
    alcDestroyContext(_context);
    alcCloseDevice(_devices);
}
