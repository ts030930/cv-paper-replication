# 기존의 깊은 Network의 문제점

앞서 살펴본 GoggleNet의 경우 19 layer였지만 그 이후 네트워크들은 매우 깊은 layer들로 설계됬다
하지만 Network가 깊어지면 깊어질수록 overfitting이 증가하고 error가 증가하는 경향이 보여
성능이 증가하지 않음을 확인하고 한계에 봉착했다.

<img width="681" height="239" alt="image" src="https://github.com/user-attachments/assets/ac959581-84e5-4c88-9c01-8fd565d3413e" />


주된 문제점은

1. Vanishing Gradient
2. 최적 loss 불가


# 해결법 : Residual Learning Framework 도입

<img width="555" height="294" alt="image" src="https://github.com/user-attachments/assets/19dd0bcc-6ce6-416e-9a15-5b5497348ee1" />

기존의 network에서는 목표인 H(x)를 바로 찾으려고(최적화) 노력했지만
H(X) = F(x) + x 로 두고 F(x) = H(x) - x를 학습하는 것을 목표로 하여 identity mapping에서 성능 향상을 도모했다.
즉 identitiy mapping에서 추가적인 학습이 필요하지 않아 진것을 의미한다.

