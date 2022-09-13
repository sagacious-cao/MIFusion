#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

float get_memory_usage();
float get_gpu_memory();