#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <iostream>

#define N 10

int main() {
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    int a[N], b[N], result[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, N * sizeof(int), a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, N * sizeof(int), b, 0, NULL, NULL);

    const char* kernelSource =
        "__kernel void sum(__global const int* a, __global const int* b, __global int* result) {"
        "    int id = get_global_id(0);"
        "    result[id] = a[id] + b[id];"
        "}";

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "sum", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);

    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, N * sizeof(int), result, 0, NULL, NULL);

    std::cout << "Result: ";
    for (int i = 0; i < N; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

