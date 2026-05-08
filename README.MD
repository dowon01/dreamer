### Model-Based RL (Dreamer) for CarRacing-v3

PyTorch를 활용하여 OpenAI Gym의 CarRacing 환경에서 시각 정보를 기반으로 세상의 물리 법칙을 학습하고, 가상 공간(상상) 속에서 최적의 자율주행 정책을 도출하는 모델 기반 강화학습(MBRL) 에이전트 구현 프로젝트

### Model Architecture

1. Vision State Abstraction (시각 정보 압축)

CNN Encoder & Decoder: 64x64 차원의 고차원 이미지 관측값을 저차원 잠재 벡터(Latent Vector)로 압축하여 연산 효율을 극대화하고, 이를 다시 원본 이미지로 복원하여 시각적 특징 추출 능력을 검증

2. Sequence Modeling (세계 모델링 및 미래 예측)

RSSM (Recurrent State Space Model): GRU 기반의 결정론적(Deterministic) 상태와 확률적(Stochastic) 상태를 결합하여, 과거 프레임과 액션 데이터를 바탕으로 환경의 관성, 마찰력 등 물리적 미래 상태와 보상을 정확하게 예측(상상)

3. Behavior Learning (행동 정책 학습)

Actor-Critic Network: World Model이 상상한 가상의 궤적 속에서 Actor 신경망은 [조향, 가속, 감속] 3-Pedal의 연속적인 액션을 도출하고, Critic 신경망은 해당 상태의 가치를 평가하여 정책을 최적화

---------------------------------------------------------------

### Result

![결과 영상](./videos/eval_run_10000.mp4)
