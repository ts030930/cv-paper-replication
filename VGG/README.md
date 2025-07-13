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



## Configuration

모델은 A~E까지 있으며 E로 갈수록 Conv Layer의 depth를 증가시킴

첫 Layer의 채널 수는 64개로 시작해서 Max Pooling을 적용 이후마다 채널을 2배 해준다.

## Discussion

(3,3) receptive field 2번과 (5,5) receptivr field 1번은 receptive field의 크기의 관점에서 동일하다.

이 부분을 좀 더 자세히 살표보면, 앞 케이스의 경우 param 수가 2x3x3으로 18이지만 뒤 케이스의 경우 25로 더 많다.
그리고 ReLU와 같은 비선형 함수를 더 많이 적용가능하고 유연하게 적용가능하다.


<img width="878" height="1005" alt="image" src="https://github.com/user-attachments/assets/4ba1a367-7c3c-41c9-bb80-da90fdd69988" />


### VGG Net이 3x3 receptive field를 사용하는 이유

1. ReLu를 여러번 적용하여 decision func(의사결정 함수)의 능력을 향상 시킴
2.
3×3 conv 3개를 쌓은 경우:

3 × (3 × 3) × C² = 27C²

7×7 conv 1개를 사용하는 경우:

7 × 7 × C² = 49C²

- 3×3 conv 3개: 27C²  
- 7×7 conv 1개: 49C² → 약 81% 더 많음  
- 비율: 49C² / 27C² ≈ 1.81

### 특수한 1 x 1 Conv Layer

입력 채널에 대한 선형변환으로 사용가능하다.
처리 후에 receptive field의 변화가 없으므로
receptive field의 변화 없이 activation func를 적용하기 위해 사용한다.

