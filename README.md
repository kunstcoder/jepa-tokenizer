# JEPA Tokenizer (Original-Code-Oriented Rebuild)

이 저장소는 `gioannides/Density-Adaptive-JEPA`의 **코드 구조/인터페이스를 최대한 따라가는 방향**으로 재구성한 실행판입니다.

## 반영 우선순위
1. 논문 텍스트보다 **오리지널 공개 코드 재활용**
2. 원본의 stage naming/유틸 함수 naming 유지 (`train_jepa`, `train_decoder`, `fsq_pack_indices` 등)
3. 단일 스크립트형 실행 UX 유지 + 2-stage JEPA→FSQ→Decoder 파이프라인 강화

## 포함 파일
- `src/jepa_tokenizer.py`: 원본 스크립트 스타일 2-stage + infer CLI (EMA target encoder, Conformer stack, FSQ 토큰 패킹 포함)
- `ANALYSIS.md`: 원본 코드 반영 관점의 정합성 검토
- `DATA_ACQUISITION.md`: LibriLight 확보/검증/JSONL 변환 가이드
- `WORK_PLAN.md`: 구현 계획 및 진행 로그
- `scripts/make_dummy_jsonl.py`: 테스트 JSONL 생성

---

## 1) 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
# 선택(실오디오 로드)
pip install torchaudio
```

## 2) 데이터 준비
```bash
python scripts/make_dummy_jsonl.py --out data/train.jsonl --count 32
```

## 3) Stage 1 (JEPA)
```bash
python src/jepa_tokenizer.py \
  --stage train_jepa \
  --jsonl data/train.jsonl \
  --out_dir outputs \
  --max_steps 20 \
  --batch_size 4 \
  --mask_ratio 0.5 \
  --device cpu
```
산출물 예시: `outputs/stage1_jepa_step20.pt` (`save_every_steps`마다 step suffix로 저장)

## 4) Stage 2 (Decoder)
```bash
python src/jepa_tokenizer.py \
  --stage train_decoder \
  --jsonl data/train.jsonl \
  --out_dir outputs \
  --stage1_ckpt outputs/stage1_jepa_step20.pt \
  --max_steps 20 \
  --batch_size 4 \
  --device cpu
```
산출물 예시: `outputs/stage2_decoder_step20.pt` (`save_every_steps`마다 step suffix로 저장)

## 5) Infer
```bash
python src/jepa_tokenizer.py \
  --stage infer \
  --jsonl data/train.jsonl \
  --ckpt outputs/stage2_decoder_step20.pt \
  --out_dir outputs/infer \
  --device cpu
```
산출물:
- `outputs/infer/recon.pt`
- `outputs/infer/indices.pt`
- `outputs/infer/packed_tokens.pt`

## 비고
- 본 구현은 원본의 명명/흐름을 우선 반영하면서, JEPA 학습(online/target encoder + mask predictor)과 decoder 학습(encoder freeze + FSQ + spectral loss)을 강화했습니다.
- DeepSpeed 기반 분산 학습 스크립트까지 완전히 동일하게 복제하지는 않고, 단일 프로세스 재현 가능한 형태로 정리했습니다.
