# JEPA Tokenizer (Reproducible Baseline)

논문(ArXiv:2512.07168)과 공개 코드(Density-Adaptive-JEPA)를 기준으로,
**학습/추론이 실제로 가능한 최소 재현 베이스라인**을 제공하는 저장소입니다.

## 포함 내용
- `ANALYSIS.md`: 논문-코드 정합성 검토 보고서
- `WORK_PLAN.md`: 작업 계획서 및 진행 상태
- `src/jepa_tokenizer.py`: Stage1/Stage2 학습 + 추론 CLI
- `scripts/make_dummy_jsonl.py`: 더미 JSONL 데이터 생성 유틸

---

## 1) 환경 준비
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
# 선택: 실제 오디오 로딩 사용 시
pip install torchaudio
```

## 1.5) 논문 데이터 확보 가이드
논문(4.1 Dataset) 기준 **LibriLight** 확보/검증 절차는 `DATA_ACQUISITION.md`를 참고하세요.

## 2) 데이터 준비
학습 스크립트는 JSONL(`{"path": "/abs/or/rel/path.wav"}`)을 받습니다.

### 더미 데이터(스모크 테스트용)
```bash
python scripts/make_dummy_jsonl.py --out data/train.jsonl --count 32
```

### 실제 데이터 예시
```json
{"path": "/data/audio/a.wav"}
{"path": "/data/audio/b.wav"}
```

---

## 3) Stage 1 학습 (JEPA-style masked prediction)
```bash
python src/jepa_tokenizer.py train_stage1 \
  --jsonl data/train.jsonl \
  --out_dir outputs \
  --max_steps 20 \
  --batch_size 4 \
  --mask_ratio 0.5 \
  --device cpu
```
산출물:
- `outputs/stage1.pt`

## 4) Stage 2 학습 (Frozen encoder + FSQ + decoder)
```bash
python src/jepa_tokenizer.py train_stage2 \
  --jsonl data/train.jsonl \
  --out_dir outputs \
  --stage1_ckpt outputs/stage1.pt \
  --max_steps 20 \
  --batch_size 4 \
  --device cpu
```
산출물:
- `outputs/stage2.pt`

## 5) 추론 (토큰화 + 복원)
```bash
python src/jepa_tokenizer.py infer \
  --jsonl data/train.jsonl \
  --ckpt outputs/stage2.pt \
  --out_dir outputs/infer \
  --device cpu
```
산출물:
- `outputs/infer/tokens.pt` (토큰 인덱스)
- `outputs/infer/recon.pt` (복원 waveform tensor)

---

## 구현 설명 (요약)
- **DAAM**: Gaussian mixture 기반의 confidence gating으로 latent sequence를 동적으로 강조.
- **Stage1**: 마스킹된 latent를 predictor로 예측(MSE).
- **Stage2**: encoder freeze 후 FSQ quantization + decoder 복원(L1).
- **Inference**: `encode_tokens` + `reconstruct` 결과 저장.

## 한계
- 본 구현은 재현성/가독성을 위한 baseline이며, 논문 수치(예: 47.5 tokens/sec)의 완전 재현을 목표로 하지는 않습니다.
- adversarial vocoder 및 분산학습(DeepSpeed) 파트는 경량화 버전으로 대체했습니다.

## 권장 확장
- 멀티스케일 STFT + GAN loss 추가
- 분산학습 및 mixed precision 추가
- 토큰 packing/bitrate 분석 리포트 자동화
