# ImageNet Classification with Deep Convolutional Neural Networks

## Architecture
<img width="1460" height="449" alt="image" src="https://github.com/user-attachments/assets/e577e8e6-b786-4346-83b3-955be51275cf" />

Input Layer : (3,224,224)

Conv 1 Layer : kernel =  (11,11), out_channel = 96, stride = 4, padding = 2

LRN + Relu Layer

Max pooling Layer : kernel =  (3,3), stride = 2 

Conv 2 Layer : kernel =  (5,5), out_channel = 256, stride = 1, padding = 2

LRN + Relu Layer

Max pooling Layer : kernel =  (3,3), stride = 2 

Conv 3 Layer : kernel =  (3,3), out_channel = 384, stride = 1, padding = 1

Relu Layer

Conv 4 Layer : kernel =  (3,3), out_channel = 384, stride = 1, padding = 1

Relu Layer

Conv 5 Layer : kernel =  (3,3), out_channel = 256, stride = 1, padding = 1

Relu Layer

Max Pooling Layer : kernel =  (3,3), stride = 2 

Fc 1 Layer : in_features = 6*6*256 (Apply Max Pooling on 13*13*256), out_features = 4096

Fc 2 Layer : in_features = 4096, out_features = 4096

Fc 3 Layer : in_features = 4096, out_features = 1000



