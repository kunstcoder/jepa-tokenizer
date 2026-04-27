# JEPA Tokenizer (Original-Code-Oriented Rebuild)

이 저장소는 `gioannides/Density-Adaptive-JEPA`의 **코드 구조/인터페이스를 최대한 따라가는 방향**으로 재구성한 경량 실행판입니다.

## 반영 우선순위
1. 논문 텍스트보다 **오리지널 공개 코드 재활용**
2. 원본의 stage naming/유틸 함수 naming 유지 (`train_jepa`, `train_decoder`, `fsq_pack_indices` 등)
3. 단일 스크립트형 실행 UX 유지

## 포함 파일
- `src/jepa_tokenizer.py`: 원본 스크립트 스타일 2-stage + infer CLI
- `ANALYSIS.md`: 원본 코드 반영 관점의 정합성 검토
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
산출물: `outputs/stage1_jepa.pt`

## 4) Stage 2 (Decoder)
```bash
python src/jepa_tokenizer.py \
  --stage train_decoder \
  --jsonl data/train.jsonl \
  --out_dir outputs \
  --stage1_ckpt outputs/stage1_jepa.pt \
  --max_steps 20 \
  --batch_size 4 \
  --device cpu
```
산출물: `outputs/stage2_decoder.pt`

## 5) Infer
```bash
python src/jepa_tokenizer.py \
  --stage infer \
  --jsonl data/train.jsonl \
  --ckpt outputs/stage2_decoder.pt \
  --out_dir outputs/infer \
  --device cpu
```
산출물:
- `outputs/infer/recon.pt`
- `outputs/infer/indices.pt`
- `outputs/infer/packed_tokens.pt`

## 비고
- 원본 저장소는 현재 접근 시 스크립트가 line-wrapped 되지 않은 형태로 배포되어 코드 감사 난이도가 높았습니다.
- 본 구현은 원본의 명명/흐름을 우선 반영하되, 로컬 단일 GPU/CPU에서도 테스트 가능한 경량화로 조정했습니다.
