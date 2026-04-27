# Density-Adaptive-JEPA 원본 코드 반영 검토 (2026-04-27)

## 요청 해석
사용자 요청의 핵심은 **"논문 충실성"보다 "오리지널 코드 재활용"**입니다.

## 원본 저장소 확인 결과
대상: `https://github.com/gioannides/Density-Adaptive-JEPA`

확인된 파일/구조:
- `train_fsqvae_jepa.py`
- `ds_ckpt_to_pt.py`
- README 상 2-stage pipeline (`train_jepa`, `train_decoder`)와 DeepSpeed 중심 실행 인터페이스

추가 관찰:
- 원본의 핵심 유틸/함수명은 다음이 공개적으로 확인됨.
  - `print_model_stats`
  - `_fsq_dim_radices`
  - `fsq_pack_indices`
  - `fsq_token_stats_from_indices`
  - `create_jepa_mask`
- README 설명 기준 핵심 구성은 GAATN(=Gaussian Adaptive Attention), FSQ, Conformer block, HiFi-GAN decoder, 2-stage 학습 흐름.

## 기존 구현 대비 문제점
기존 로컬 구현은 동작은 가능했지만,
- CLI가 `train_stage1`/`train_stage2` 서브커맨드 기반으로 원본과 인터페이스 괴리,
- 원본 유틸 함수명/토큰 패킹 유틸 반영 부족,
- 원본 흐름(`--stage train_jepa/train_decoder`)과 거리가 있었습니다.

## 재구현 반영 사항
`src/jepa_tokenizer.py`를 원본 지향으로 재정렬했습니다.

1. 인터페이스 정렬
- 단일 엔트리 + `--stage {train_jepa, train_decoder, infer}`

2. 원본 유틸 함수명 재사용
- `print_model_stats`
- `_fsq_dim_radices`
- `fsq_pack_indices`
- `fsq_token_stats_from_indices`
- `create_jepa_mask`

3. 모델 구성 명명/흐름 정렬
- `GaussianAdaptiveAttention`(GAATN 대응)
- `TinyConformerBlock` (경량 conformer 대응)
- `JEPAEncoder`, `JEPAPredictor`, `FSQ`, `SimpleHiFiGenerator`
- Stage 1: masked JEPA latent prediction
- Stage 2: encoder freeze + FSQ + decoder + spectral 보조 loss

4. 토큰 통계/패킹 경로 반영
- decoder 학습 로그에서 tokens/sec 출력
- infer 시 packed token tensor 저장

## 결론
현재 버전은 논문 텍스트 중심 구현에서 벗어나,
**원본 저장소의 함수명/실행 흐름/토큰 패킹 유틸을 우선 재사용한 구조**로 업데이트되었습니다.
