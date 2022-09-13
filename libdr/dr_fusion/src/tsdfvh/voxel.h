// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Struct representing a voxel.
 */
struct Voxel {
  /** Signed distance function */
  float sdf;

  /** Color */
  uchar3 color;
//   uchar3 instance_color;
  /** Accumulated SDF weight */
  unsigned char weight;

  unsigned char  InstanceIndex = 0;
  unsigned int InstanceFrameNum = 0;
  /**
   * @brief      Combine the voxel with a given one
   *
   * @param[in]  voxel       The voxel to be combined with
   * @param[in]  max_weight  The maximum weight
   */
  __host__ __device__ void Combine(const Voxel& voxel,
                                   unsigned char max_weight) {
    color.x = static_cast<unsigned char>(
        (static_cast<float>(color.x) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.x) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) +
        static_cast<float>(voxel.weight)));
    color.y = static_cast<unsigned char>(
        (static_cast<float>(color.y) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.y) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) +
        static_cast<float>(voxel.weight)));
    color.z = static_cast<unsigned char>(
        (static_cast<float>(color.z) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.z) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) +
        static_cast<float>(voxel.weight)));

    sdf = (sdf * static_cast<float>(weight) +
          voxel.sdf * static_cast<float>(voxel.weight)) /
              (static_cast<float>(weight) + static_cast<float>(voxel.weight));

    weight = weight + voxel.weight;
    if (weight > max_weight) weight = max_weight;
  }

__host__ __device__ void SemanticCombine(const Voxel& voxel,
                                   unsigned char max_weight,unsigned char* instanceColorList_gpu) {

    color.x = static_cast<unsigned char>(
        (static_cast<float>(color.x) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.x) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) + static_cast<float>(voxel.weight)));
    
    color.y = static_cast<unsigned char>(
        (static_cast<float>(color.y) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.y) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) + static_cast<float>(voxel.weight)));
    
    color.z = static_cast<unsigned char>(
        (static_cast<float>(color.z) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.z) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) + static_cast<float>(voxel.weight)));                            

    if(InstanceIndex == 0 && InstanceFrameNum == 0)
    {
        if(voxel.InstanceIndex !=0)
        {
            InstanceIndex = voxel.InstanceIndex;
            InstanceFrameNum = 1;
        }
        // else
        // {
        //     instance_color = make_uchar3(200,200,200);
        // }
    }
    else if(InstanceIndex !=0 && voxel.InstanceIndex!=0)
    {
        if(InstanceIndex == voxel.InstanceIndex)
        {
            InstanceFrameNum+=1;
        }
        else
        {
            if(InstanceFrameNum > 1 && InstanceFrameNum < 3)
            {
                InstanceFrameNum-=1;
            }
            if(InstanceFrameNum==0)
            {
                InstanceIndex = 0;
            }
        }
    }
    else if(InstanceIndex !=0 && voxel.InstanceIndex==0)
    {


    }

    sdf = (sdf * static_cast<float>(weight) + voxel.sdf * static_cast<float>(voxel.weight)) / (static_cast<float>(weight) + static_cast<float>(voxel.weight));

    weight = weight + voxel.weight;
    if (weight > max_weight) 
        weight = max_weight;
  }

//   __host__ __device__ void SemanticCombine(const Voxel& voxel,unsigned char max_weight,unsigned int* classColorList_gpu) 
//   {
//     // printf("enter combine\n");
//     // voxel.isSemanticColored = true;
//     // isSemanticColored = true;
//     if((false == voxel.isSemanticColored)&&false == isSemanticColored)
//     {
//       color = make_uchar3(200,200,200);
//     }
//     else
//     {
//       isSemanticColored = true;
//       for(int score_idx = 0;score_idx<81;score_idx++)
//       {
//         score[score_idx]+= voxel.score[score_idx];
//       }
//       int Semanticindex = getMaxScoreIndex();
//       if(Semanticindex != -1)
//       {
//         // printf("%d\n",Semanticindex);
//         color.x = classColorList_gpu[3*Semanticindex+0];
//         color.y = classColorList_gpu[3*Semanticindex+1];
//         color.z = classColorList_gpu[3*Semanticindex+2];
//       }
//       else
//       {
//         isSemanticColored = false;
//         color = make_uchar3(0,255,0);
//       }
//     }
//     sdf = (sdf * static_cast<float>(weight) + voxel.sdf * static_cast<float>(voxel.weight)) /(static_cast<float>(weight) + static_cast<float>(voxel.weight));
//     weight = weight + voxel.weight;
//     if (weight > max_weight) 
//       weight = max_weight;
//     // printf("leave combine\n");
//   }



};

  



}  // namespace tsdfvh

}  // namespace refusion
