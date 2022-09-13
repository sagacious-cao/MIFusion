#include "memoryManager.h"



float  get_memory_usage()
{
    FILE*file=fopen("/proc/meminfo","r");
    if(file == NULL)
    {
        fprintf(stderr,"cannot open /proc/meminfo\n");
        return -1;
    }
    char keyword[20];
    char valuech[20];
    float mem        =0;
    float free_mem   =0;
    fscanf(file,"MemTotal: %s kB\n",keyword);
    mem=atof(keyword)/1024;
    
    fscanf(file,"MemFree: %s kB\n",valuech);
    fscanf(file,"MemAvailable: %s kB\n",valuech);
    
    free_mem=atof(valuech)/1024;
    fclose(file);
    fprintf(stderr,"Memory %.2f GB / %.2f GB.\n", (mem - free_mem)/1024,mem/1024);
    return mem - free_mem;
}


float get_gpu_memory()
{
    int nCudaNums = 0;
    size_t avail(0);//可用显存
    size_t total(0);//总显存
    cudaGetDeviceCount(&nCudaNums);//获取显卡数量
    for (int i = 0 ;i<nCudaNums;i++)
    {
        cudaSetDevice(i);
        cudaMemGetInfo(&avail,&total);
        printf("GPU avail: %.2f GB / %.2f GB\n",(total - avail)*1.0/1024/1024,total*1.0/1024/1024);
        //可以在这儿输出  
    }
    return 0;
}