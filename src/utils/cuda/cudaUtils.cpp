
#include "cudaUtils.hpp"

void CudaUtils::printCudaDevices(std::ostream &outputStream) {

	int nDevices;
	char buffer[100];

	cudaGetDeviceCount(&nDevices);
	outputStream << "==== CUDA DEVICES ====" << std::endl;
	outputStream << "Found " << nDevices << " devices !" << std::endl;
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		outputStream << "Device Number: " << i << std::endl;
		outputStream << "\tDevice name:                   "  << prop.name << std::endl;
		outputStream << "\tPCI Device:                    " 
			<< prop.pciBusID << ":" << prop.pciDeviceID << ":" << prop.pciDomainID << std::endl;
		outputStream << "\tMajor revision number:         " << prop.major << std::endl;
		outputStream << "\tMinor revision number:         " <<   prop.minor << std::endl;
		outputStream << "\tMemory Clock Rate :            " << prop.memoryClockRate/1000 << " MHz" << std::endl;
		outputStream << "\tMemory Bus Width:              " << prop.memoryBusWidth << " bits" << std::endl;
		outputStream << "\tPeak Memory Bandwidth:         " 
			<< 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
		outputStream << "\tTotal global memory:           " <<   prop.totalGlobalMem/(1024*1024) << " MB" << std::endl;
		outputStream << "\tTotal shared memory per block: " <<   prop.sharedMemPerBlock/1024 << " kB" << std::endl;
		outputStream << "\tTotal registers per block:     " <<   prop.regsPerBlock/1024 << " kB" << std::endl;
		outputStream << "\tTotal constant memory:         " <<   prop.totalConstMem/1024 << " kB" << std::endl;
		outputStream << "\tMaximum memory pitch:          " <<   prop.memPitch/(1024*1024) << " MB" << std::endl;
		outputStream << "\tNumber of multiprocessors:     " <<   prop.multiProcessorCount << std::endl;
		outputStream << "\tMaximum threads per SM:        " <<   prop.maxThreadsPerMultiProcessor << std::endl;
		outputStream << "\tMaximum threads per block:     " <<   prop.maxThreadsPerBlock << std::endl;

		sprintf(buffer, "%ix%ix%i", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		outputStream << "\tMaximum thread block dimension " <<  buffer << std::endl;
		sprintf(buffer, "%ix%ix%i", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		outputStream << "\tMaximum grid dimension         " <<  buffer << std::endl;
		outputStream << "\tWarp size:                     " <<   prop.warpSize << std::endl;
		outputStream << "\tTexture alignment:             " <<   prop.textureAlignment << std::endl;
		outputStream << "\tTexture picth alignment:       " <<   prop.texturePitchAlignment << std::endl;
		outputStream << "\tSurface alignment:             " <<   prop.surfaceAlignment << std::endl;
		outputStream << "\tConcurrent copy and execution: " <<   (prop.deviceOverlap ? "Yes" : "No") << std::endl;
		outputStream << "\tKernel execution timeout:      " <<   (prop.kernelExecTimeoutEnabled ?"Yes" : "No") << std::endl;
		outputStream << "\tDevice has ECC support:        " <<   (prop.ECCEnabled ?"Yes" : "No") << std::endl;
		outputStream << "\tCompute mode:                  " 
			<<   (prop.computeMode == 0 ? "Default" : prop.computeMode == 1 ? "Exclusive" :
					prop.computeMode == 2 ? "Prohibited" : "Exlusive Process") << std::endl;
	}

	outputStream << "======================" << std::endl;
}

void CudaUtils::logCudaDevices(log4cpp::Category &log_output) {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	char buffer[100];
	log_output.infoStream() << "==== CUDA DEVICES ====";
	log_output.infoStream() << "Found " << nDevices << " devices !";
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		log_output.infoStream() << "Device Number: " << i;
		log_output.infoStream() << "\tDevice name:                   "  << prop.name;
		log_output.infoStream() << "\tPCI Device:                    " 
			<< prop.pciBusID << ":" << prop.pciDeviceID << ":" << prop.pciDomainID;
		log_output.infoStream() << "\tMajor revision number:         " << prop.major;
		log_output.infoStream() << "\tMinor revision number:         " <<   prop.minor;
		log_output.infoStream() << "\tMemory Clock Rate :            " << prop.memoryClockRate/1000 << " MHz";
		log_output.infoStream() << "\tMemory Bus Width:              " << prop.memoryBusWidth << " bits";
		log_output.infoStream() << "\tPeak Memory Bandwidth:         " 
			<< 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << " GB/s";
		log_output.infoStream() << "\tTotal global memory:           " <<   prop.totalGlobalMem/(1024*1024) << " MB";
		log_output.infoStream() << "\tTotal shared memory per block: " <<   prop.sharedMemPerBlock/1024 << " kB";
		log_output.infoStream() << "\tTotal registers per block:     " <<   prop.regsPerBlock/1024 << " kB";
		log_output.infoStream() << "\tTotal constant memory:         " <<   prop.totalConstMem/1024 << " kB";
		log_output.infoStream() << "\tMaximum memory pitch:          " <<   prop.memPitch/(1024*1024) << " MB";
		log_output.infoStream() << "\tNumber of multiprocessors:     " <<   prop.multiProcessorCount;
		log_output.infoStream() << "\tMaximum threads per SM:        " <<   prop.maxThreadsPerMultiProcessor;
		log_output.infoStream() << "\tMaximum threads per block:     " <<   prop.maxThreadsPerBlock;

		sprintf(buffer, "%ix%ix%i", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		log_output.infoStream() << "\tMaximum thread block dimension " <<  buffer;
		sprintf(buffer, "%ix%ix%i", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		log_output.infoStream() << "\tMaximum grid dimension         " <<  buffer;
		log_output.infoStream() << "\tWarp size:                     " <<   prop.warpSize;
		log_output.infoStream() << "\tTexture alignment:             " <<   prop.textureAlignment;
		log_output.infoStream() << "\tTexture picth alignment:       " <<   prop.texturePitchAlignment;
		log_output.infoStream() << "\tSurface alignment:             " <<   prop.surfaceAlignment;
		log_output.infoStream() << "\tConcurrent copy and execution: " <<   (prop.deviceOverlap ? "Yes" : "No");
		log_output.infoStream() << "\tKernel execution timeout:      " <<   (prop.kernelExecTimeoutEnabled ?"Yes" : "No");
		log_output.infoStream() << "\tDevice has ECC support:        " <<   (prop.ECCEnabled ?"Yes" : "No");
		log_output.infoStream() << "\tCompute mode:                  " 
			<<   (prop.computeMode == 0 ? "Default" : prop.computeMode == 1 ? "Exclusive" :
					prop.computeMode == 2 ? "Prohibited" : "Exlusive Process");
	}

	log_output.infoStream() << "======================";
}



