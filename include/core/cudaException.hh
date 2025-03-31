#pragma once

#include <exception>
#include <iostream>
#include <cuda_runtime.h>

class CudaException : public std::exception{
    private:
        const char* msg;
    
    public:
        CudaException(const char* msg): msg(msg){}
        virtual const char* what() const throw(){
            return msg;
        }
    static void throwIfCUDAErrorsOccurred(const char* msg){
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess){
            std::cerr << error << ": " << msg;
            throw CudaException(msg);
        }
    }
};