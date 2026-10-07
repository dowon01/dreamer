# Dreamer for CarRacing-v3

PyTorch로 구현한 모델 기반 강화학습(Dreamer) 에이전트
카메라 화면만 보고 환경의 동작 방식을 학습한 **World Model** 안에서 주행을 "상상"하며 정책을 학습하고, Gymnasium `CarRacing-v3`에서 **새 트랙 20개 평균 903점**(해결 기준 900점)을 달성함.

| 실제 주행 결과 | 상상 속 주행 (화면 없이 World Model 안에서 Actor가 운전) |
|:---:|:---:|
| <img src="videos/eval_run_best.gif" width="380"> | <img src="videos/dream_imagine.gif" width="300"> |

## Architecture

```
64x64 화면 → CNN Encoder → RSSM (결정적 h: GRU 512 + 확률적 z: 64×32 categorical)
                                │
                                ├─ Decoder / Reward / Continue 예측
                                │
                                └─ 상상 주행 (관측 없이 15 step) → Actor · Critic 학습
```

**World Model**
- **CNN Encoder / Decoder**: 64x64 관측을 1024차원 임베딩으로 압축하고, 잠재 상태에서 다시 이미지를 복원
- **RSSM**: GRU 기반 결정적 상태(h)와 이산 확률 상태(z, Straight-Through 샘플링)를 결합해 다음 상태를 예측. 관측이 있으면 posterior, 없으면 prior로 미래를 상상
- **Reward / Continue head**: 각 상태에서 받을 보상과 에피소드 지속 여부를 예측
- 학습: 이미지 복원 + 보상·종료 예측 + KL balancing(0.8, free bits)

**Behavior (Actor-Critic)**
- 실제 데이터에서 얻은 상태를 출발점으로 World Model 안에서 15 step을 상상하고, λ-return(γ=0.99, λ=0.95)으로 학습
- **Actor**: 조향·가속·브레이크 3차원 연속 행동. tanh 평균 + std∈[0.1, 1] 정규분포를 [-1, 1]로 clip하며, **REINFORCE**(advantage = (λ-return − V) / 수익 범위)로 학습
- **Critic**: 상태 가치 예측, 느리게 따라가는 target critic으로 λ-return을 계산

## 구현하며 해결한 문제

| 문제 | 원인 | 해결 |
|---|---|---|
| 차가 멈추거나 조작이 ±1로 굳는 붕괴가 반복됨 | World Model과 Critic을 미분해 Actor를 학습(dynamics backprop)하면, 부정확한 가치 기울기가 보상 기울기를 뒤집음 (진단 스크립트로 확인) | Actor 학습을 **REINFORCE + 수익 정규화**(DreamerV3 방식)로 전환 |
| 정책이 tanh 끝값으로 포화 | tanh 변환 분포는 엔트로피를 샘플로만 추정 가능해 탐험이 유지되지 않음 | clip 정규분포 + 해석적 엔트로피 |
| 보상·종료 예측이 한 스텝 어긋남 | 행동의 결과를 행동 전 상태로 예측하도록 데이터가 정렬됨 | "행동 → 도달한 상태 → 보상" 순으로 데이터 형식 재설계 |

## Results

- **새 트랙 20개 평가**: 평균 **903.2 ± 26.4**, 최저 866 (실패 트랙 없음)
- **학습 속도**: 초기 구현은 약 1만 iteration(8일) 학습 후 평균 약 870점 → 개선 후 **100 iteration(약 2시간)** 만에 평가 895점

| World Model 복원 (매 스텝 관측) | World Model 예측 (같은 행동, 15 step마다 관측) |
|:---:|:---:|
| <img src="videos/dream_recon.gif" width="380"> | <img src="videos/dream_compare.gif" width="380"> |

- **복원**: 평균 픽셀 오차 1.7/255로 연석·계기판까지 거의 그대로 복원
- **예측**: 관측 없이 5 step 후 오차 4.6 → 30 step 후 12.7. 화면 밖의 트랙 모양은 알 수 없어, 길게 상상할수록 실제와 다른 도로를 만들어냄
- **한계**: 급커브에서 과속해 잔디로 이탈했다가 복귀하는 경우가 있고, 상상 속 보상이 실제보다 높게 예측되는 경향이 있음
