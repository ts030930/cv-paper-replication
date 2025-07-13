# Introduction

기존 개선 시도 : small receptive field & small stride

본 paper의 핵심 기여 : conv layer를 늘림으로써 depth를 증가시킴 ( 3x3 kernel 사용으로 depth를 늘려도 parameter 크게 증가 X )


# ConvNet Configuration

## Architecture

input size : (3,224,224) RGB image

pre processig : substract each RGB mean 외 다른 전처리기법 적용 X

모든 receptive field ( filter ) : (3,3) 사용

각 Conv Layer의 stride = 1이고 padding을 통해 spatial resolution을 유지/보존 시키려고 함
즉, (3,3) filter 적용 후에는 padding = 1을 적용시켜서 보존 시킨다.

Spatial Pooling은 Max Pooling Layer((2,2), stride = 2)를 통해 진행한다.

Conv Layer 이후 3개의 FC Layer가 이어지는데,
앞 두 Layer는 4096 dimension, 마지막은 1000(dataset에서 지원하는 class 수) dimesion으로 한다.

마지막으로 Softmax Layer를 통해 각 클래스의 확률값을 도출 후 학습시킨다.

모든 hidden layer에는 ReLU를 적용한다.


### 특수한 1 x 1 filter

입력 채널에 대한 선형변환으로 사용가능하다.
처리 후 비선형 함수 적용

