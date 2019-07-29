# Face-R-FCN

Pytorch implementation of the Face R-FCN deep learning architecture for detecting faces - https://arxiv.org/abs/1709.05256
This was an attempt to completely segregate each component and write cleaner code to help others understand the individual steps behind the complex Face R-FCN architecture. 

Versions required: Pytorch 0.4.1 and CUDA 9.2

# Before running training.py or inference.py

1. Compile CUDA modules: cd models && sh make.sh
2. For training: python3 training.py 
3. For inference/validation: python3 inference.py

