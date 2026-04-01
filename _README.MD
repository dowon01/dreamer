### data
CarRacing-v3 환경 구축
64 * 64 해상도
직접 수동 운전한 20개 에피소드
리플레이 버퍼 : 시퀀스 (50프레임 단위) 학습이 가능하도록 데이터 셔플하여 공급

### model
image
↓
encoder
↓
RSSM
(h,z)
↓
decoder
reward model
continue model

그리고 actor 학습

(h,z)
↓
actor
↓
action
↓
RSSM prior rollout
↓
reward/value
↓
lambda return
↓
actor critic update

# tensorboard
tensorboard --logdir=runs