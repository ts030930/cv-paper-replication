# Introduction

Inception 개념 도입

# Motivation and High Level Considerations

network의 성능을 향상시키는 직관적인 방식 : network 크기 늘리기

그 방법으로는 

1. 계층 수 증가
2. 각 계층의 유닛수 증가

이러한 방식의 근본적인 단점 2가지는

1. 많은 파라미터 -> overfitting 증가
2. 연산량 증가

이러한 문제들을 근본적으로 해결하기 위해서는

Sparse Connection 접근

CNN의 합성곱 필터는 이러한 접근을 사용하지만,
전통적인 CNN은 GPU 병렬화를 위해 feature map들을 fully connection 방식으로 이용했다.
그러므로 공간적으로는 sparse하지만 feature dimension 관점으로는 dense하게 된것.

Inception의 아이디어

1. 각 계층에서 서로 다른 크기의 필토와 풀링을 병렬로 진행
2. 이를 하나의 출력으로 합치는 방식으로 sparse 구현
3. 1x1 합성곱을 이용해 channel 수를 줄이고 큰 필터를 사용해서 연산량 줄이기
4. 즉 실제 연결 패턴을 sparse하게 구현

즉 sparse의 효과를 dense 연산으로 흉내내기


# Architectural Details

저수준 layer에서는 입력과 가까우므로 1,1 layer로 좁은 지역에 집중하고
뒤로 갈수록 3,3이나 5,5 layer를 사용하여 넓은 지역을 이용하게 한다.
그리고 pooling을 사용한다

즉 Inception 모듈이 쌓이고 network의 상위 층으로 갈수록
추상적인 특징을 잡아내면 공간적인 집중도(spatial concentration)은 감소하게됨
즉 3,3이나 5,5 layer를 사용해야한다.

이러한 방식의 문제점은 출력 채널수가 너무 증가한다는 것이 있다.
그러므로 적절하게 차원 축소를 진행해야한다.

즉 1,1 conv layer로 차원 축소를 진행시킨다

<img width="1240" height="523" alt="image" src="https://github.com/user-attachments/assets/56c18043-2321-4c07-b6ac-4e313b684a18" />

## Configuration

모델은 A~E까지 있으며 E로 갈수록 Conv Layer의 depth를 증가시킴

첫 Layer의 채널 수는 64개로 시작해서 Max Pooling을 적용 이후마다 채널을 2배 해준다.

## Discussion

(3,3) receptive field 2번과 (5,5) receptivr field 1번은 receptive field의 크기의 관점에서 동일하다.

이 부분을 좀 더 자세히 살표보면, 앞 케이스의 경우 param 수가 2x3x3으로 18이지만 뒤 케이스의 경우 25로 더 많다.
그리고 ReLU와 같은 비선형 함수를 더 많이 적용가능하고 유연하게 적용가능하다.


<img width="878" height="1005" alt="image" src="https://github.com/user-attachments/assets/4ba1a367-7c3c-41c9-bb80-da90fdd69988" />




