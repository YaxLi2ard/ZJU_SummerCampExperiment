# ZJU_SummerCampExperiment


## 模型权重下载
实验产出的模型权重下载链接：[GoogleDrive](https://drive.google.com/drive/folders/19zOXVr5xmxtXvVQUlOEGF_MGCapKMfrG?usp=drive_link)

## 主要运行环境
```
numpy == 1.22.4
pytorch == 2.0.0
spikingjelly == 0.0.0.0.14
```

## 文件结构说明
```

ZJU_SummerCampExperiment:
│          
├─code1
│  │  // 阶段2 步骤1，在DVS128 Gesture上微调ResNet(SNN)模型
│  │  LIAF.py   // ResNet模型代码
│  │  LIAFResNet.py   // ResNet模型代码
│  │  LIAFResNet_18.py   // ResNet模型代码
│  │  main.py   // 在DVS128上训练ResNet
│  │  merging_greedy.py   // greedy soup合并模型
│  │  merging_uniform.py   // uniform soup合并模型
│  │  usetest.py   // 对指定权重在DVS128 Gesture测试集上测试
│  │  
│  ├─util 
│  │      // ResNet模型依赖代码
│  │      
│  └─weights
│          // 存放微调的模型权重
│          
├─code2
│  │  // 阶段2 步骤1，在DVS128 Gesture上微调Spike-DrivenV2模型
│  │  main.py   // 在DVS128上训练Spike-DrivenV2
│  │  merging_greedy.py   // greedy soup合并模型
│  │  merging_uniform.py   // uniform soup合并模型
│  │  models.py   // Spike-DrivenV2模型代码
│  │  usetest.py   // 对指定权重在DVS128 Gesture测试集上测试
│  │  
│  └─weights
│          // 存放微调的模型权重
│
├─code2_new 
│  │  // 重新进行的 阶段2 步骤1，在DVS128 Gesture上微调Spike-DrivenV2模型
│  │  main.py   // 在DVS128上训练Spike-DrivenV2
│  │  merging_greedy.py   // greedy soup合并模型
│  │  merging_uniform.py   // uniform soup合并模型
│  │  check.py   // 简单查找合并后会使准确率降低的权重（无效合并权重）
│  │  flag.npy   // 记录有效合并权重的序号，由check.py产生
│  │  log.txt   // 部分训练过程的输出记录
│  │  models.py   // Spike-DrivenV2模型代码
│  │  usetest.py   // 对指定权重在DVS128 Gesture测试集上测试
│  │  
│  └─weights
│          // 存放微调的模型权重
│
├─code3
│  │  // 阶段2 步骤2，在常规静态数据集CIFAR-10上微调Spike-DrivenV2模型
│  │  main.py   // 在CIFAR10上训练Spike-DrivenV2
│  │  merging_greedy.py   // greedy soup合并模型
│  │  merging_uniform.py   // uniform soup合并模型
│  │  models.py   // Spike-DrivenV2模型代码
│  │  usetest.py   // 对指定权重在CIFAR-10测试集上测试
│  │  
│  ├─dataset
│  │      // 放CIFAR-10数据集
│  │      
│  └─weights
│          // 存放微调的模型权重
│          
├─DVS128
│      // 存放下载的DVS128 Gesture数据集
│      
└─ESI_lt
    │  // 阶段1，减小ES_ImageNet数据集规模，并在其上预训练ResNet(SNN)和Spike-DrivenV2
    │  delete_file.py   // 删除数据集中的文件、文件夹，缩减数据集规模
    │  delete_label.py   // 缩减数据集规模后，删除txt中的多余标签
    │  LIAF.py   // ResNet模型代码
    │  LIAFResNet.py   // ResNet模型代码
    │  LIAFResNet_18.py   // ResNet模型代码
    │  main1.py   // ResNet在小型ES_ImageNet上预训练
    │  main2.py   // Spike-DrivenV2在小型ES_ImageNet上预训练
    │  models.py   // Spike-DrivenV2模型代码
    │  process.py   // 将数据流积分为帧数据，存于processed文件夹中
    │  processed_dataset.py   // 用于帧数据加载的dataset类
    │  query.py   // 查询文件总数
    │
    ├─processed
    │      // 存放处理后的帧数据
    │
    ├─ES-imagenet-0.18
    │      // 存放解压后的ES_ImageNet数据集
    │
    └─util 
            // ResNet模型依赖代码
            

```
  
