# Imagenet_classification
# **Comparative Analysis of Residual vs. Plain Networks on CIFAR-10**  

## ** Project Overview**  
This project implements and compares **Residual Networks (ResNets)** and **Plain Networks (PlainNets)** on the **CIFAR-10 dataset**, analyzing how residual connections impact model performance, optimization stability, and scalability with depth.  

### **Key Findings**  
**ResNets outperform PlainNets**, especially in deeper architectures (e.g., ResNet110: **86%** vs. PlainNet110: **20%**).  
**Skip connections enable stable training**—PlainNets fail completely at 110 layers due to vanishing gradients.  
**Visualized loss landscapes** confirm smoother optimization paths for ResNets.  

---

## **🛠 Model Architectures**  
### **Implemented Models**  
| Model       | Depth | Parameters per Block | Key Feature                     |
|-------------|-------|----------------------|---------------------------------|
| ResNet20    | 20    | 6.5K                 | Skip connections, batch norm    |
| ResNet56    | 56    | 18K                  |                                 |
| ResNet110   | 110   | 36K                  | He init, batchnorm gamma=0     |
| PlainNet20  | 20    | 6.5K                 | No skip connections            |
| PlainNet56  | 56    | 18K                  |                                 |
| PlainNet110 | 110   | 36K                  | Fails at training (20% acc)    |

### **Core Components**  
- **Initial Layer**: 3x3 conv (16 channels) → BatchNorm → ReLU.  
- **Residual Block**: Two 3x3 convs + skip connection (identity or dimension-adjusted).  
- **Plain Block**: Same as residual but **no skip connections**.  
- **Classifier**: Adaptive average pooling → Linear layer.  

---

## ** Results**  
### **Performance Comparison**  
| Model       | Validation Acc | Test Loss | Test Acc |
|-------------|---------------|-----------|----------|
| ResNet20    | 0.85          | 0.6680    | 0.85     |
| ResNet56    | 0.84          | 0.6491    | 0.85     |
| ResNet110   | 0.79          | 0.9578    | 0.86     |
| PlainNet20  | 0.84          | 0.7499    | 0.83     |
| PlainNet56  | 0.79          | 1.5385    | 0.77     |
| PlainNet110 | 0.20          | 2.1026    | 0.20     |

### **Key Insights**  
- **ResNets scale better**: Accuracy remains stable (85–86%) up to 110 layers.  
- **PlainNets degrade with depth**: Accuracy drops from 83% (20 layers) to 20% (110 layers).  
- **Loss landscapes**: ResNets exhibit smoother surfaces (see `figures/`).  
---

## **🔍 References**  
- [He et al. (2016)](https://arxiv.org/abs/1512.03385): Original ResNet paper.  
- CIFAR-10 Dataset: [Official Page](https://www.cs.toronto.edu/~kriz/cifar.html).  

---
