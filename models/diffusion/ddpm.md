# DDPM/UNet TODO 체크리스트

## ✅ Must-fix (학습/샘플링에 필수)

* [x] `UNet.forward` 마지막에 **`return o`** 추가
* [x] **시간 임베딩 사용**: `e = time_embedding(t)` → `e = time_embedding_mlp(e)` → 모든 `ResBlock`에 `t_emb=e` 전달
* [x] `nn.Sequential`로는 `t_emb` 전달 불가 → \*\*`ModuleList`\*\*로 바꾸고 `forward`에서 직접 호출해 `t_emb` 넘기기
* [x] **GroupNorm 그룹 수 0 방지**: `num_groups = max(1, min(32, C))` 또는 **stem conv**로 채널을 먼저 64로 올리기
* [ ] **알파/베타 수식 수정**: `alphas = 1 - betas` (현재 `betas - 1` 오류) → `alphas_bar = cumprod(alphas)`
* [x] **버퍼 등록**: `betas, alphas, alphas_bar, sqrt_*`를 `register_buffer`로 보관 (디바이스/저장 일관성)
* [x] **학습 루프**: `optimizer.zero_grad(set_to_none=True)` 호출, `epoch_loss += loss.item()`
* [x] **검증 루프**: `t`를 `device`로 올리기, `epoch_loss += loss.item()`
* [x] `UNet.forward` 호출부에서 **(x, t, y)** 시그니처 일치시키기 (추론부 포함)

## 🎛 Conditioning & 임베딩

* [x] **클래스 임베딩**: `nn.Embedding(num_classes + 1, d)` 추가(또는 time-emb dim에 맞춤)
* [x] **임베딩 결합**: `time_emb + class_emb`(또는 별도 MLP 후 합)
* [x] **CFG 학습 반영**: train loop에서 `apply_cfg_conditioning` 사용(확률 `p`로 null token); 임베딩 테이블 크기에 **null token 포함**
* [ ] `UNet.forward` 시그니처 유연화: `def forward(self, x, t, y=None, *, cond_drop_mask=None)`

## ⛏ UNet/블록 구조

* [x] **초기 stem conv** 추가: `Conv2d(in_ch, 64, 3, padding=1)` → 이후 ResBlocks 시작
* [x] **Skip 연결 정합**: add 방식 점검 또는 **concat → ResBlocks**(표준 UNet)로 변경 검토
* [x] **SelfAttention 구현**: 현재 패스스루 → QKV, MHSA, proj\_out 구현(큰 해상도에서는 비활성/다운샘플 후 적용 고려)

## 🔢 노이즈 스케줄 & 손실

* [ ] **스케줄 옵션화**: linear 외 **cosine/sigmoid** 스케줄 추가
* [ ] **손실 옵션**: `ε`-pred 기본 + **`v`-pred 토글** 추가
* [ ] **Gradient clipping**: `clip_grad_norm_(model.parameters(), 1.0)` 등

## 🧪 학습/검증 안정화

* [ ] **AMP 도입**: `autocast()` + `GradScaler`
* [ ] **평균/로깅**: 검증에서 평균 loss 반환, step/epoch 로깅( loss, lr, grad-norm )
* [ ] **시각화**: 주기적 샘플 이미지 저장/로깅

## 🧰 샘플링(추론) 루프

* [ ] **타임 루프**: `for t in reversed(range(T))` 또는 서브스텝 인덱스 매핑
* [ ] **posterior 공식 적용**

  * [ ] $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$
  * [ ] $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\right)$
  * [ ] $x_{t-1} = \mu_\theta + \sigma_t z,\; \sigma_t=\sqrt{\tilde\beta_t}$ (t>0일 때만 `z∼N(0,I)`)
* [ ] **CFG 추론**: `eps = eps_uncond + s * (eps_cond - eps_uncond)`
* [ ] **steps ≠ T 지원**: 서브스케줄 인덱스(예: `np.linspace`)로 t 매핑
* [ ] **인덱스/디바이스 안전성**: 텐서 인덱싱 shape 맞추기, `device` 일관 유지

## 💾 체크포인트 & 재현성

* [ ] **체크포인트 확대**: 모델 + 옵티마 + 스케줄러 + 스텝/에폭 + EMA 상태 저장
* [ ] **EMA 파라미터**: decay 0.999–0.9999
* [ ] **Seed 고정**: `torch`, `numpy`, `random` + `cudnn.deterministic=True, benchmark=False`

## 📐 초기화/정규화

* [ ] **Kaiming 설정 재검토**: SiLU에 맞춘 파라미터(대체로 fan-in/‘relu’로 OK) 일관화
* [ ] **GroupNorm 약수 보장**: 채널의 약수로 그룹 수 선택(32→16→8→4→1 등 폴백)

## 🧾 모듈 I/O 계약

* [ ] `SinusoidalPositionalEmbedding.forward(t)->(B,d)` 규격 고정(dtype/device 포함)
* [ ] `UNet.forward` 입력/출력(조건/무조건, 반환 `eps_pred`) **docstring**으로 명시

## ✨ 옵션(품질/개발 편의)

* [ ] `torch.compile`(PyTorch 2.x)로 훈련/추론 가속
* [ ] TensorBoard/W\&B 연동
* [ ] **DDIM/PLMS/Heun** 등 대체 샘플러 옵션 추가
* [ ] 주기적 **EMA-모델로 샘플링**하여 품질/안정성 모니터링
