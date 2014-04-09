
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_

#include <ostream>
#include <log4cpp/Category.hh>
#include "cuda.h"
#include "cuda_runtime.h"

#include "log.h"


#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }

class CudaUtils {

	public:
		static void printCudaDevices(std::ostream &outputStream);
		static void logCudaDevices(log4cpp::Category &log_output);
		
};

inline void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort = true) {
	if (code != cudaSuccess) 
	{
		log_console.errorStream() << "GPU Assert => " << cudaGetErrorString(code) << " in file " <<  file << ":" << line << ".";
		if (abort) 
			throw std::logic_error("GPU Assert false !");
	}
}

inline void checkKernelExecution() {
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			log_console.errorStream() << "Kernel launch failed : " << cudaGetErrorString(error);
			throw std::logic_error("Kernel launch failed !");
		}
}

#endif
