# 논문/공개코드 구현 검토 보고서

## 검토 대상
- 논문: **JEPA as a Neural Tokenizer: Learning Robust Speech Representations with Density Adaptive Attention** (arXiv:2512.07168, 2025-12-08 제출)
- 코드: `gioannides/Density-Adaptive-JEPA` 공개 저장소

## 핵심 주장 요약
1. 2-stage 학습 (JEPA pretrain → FSQ + vocoder decoder)
2. DAAM(밀도 적응 attention) 기반 시계열 선택
3. 저프레임레이트 토큰화(논문 초록 기준 2.5Hz / 47.5 tokens-sec)
4. 복원 가능한 neural tokenizer

## 공개 코드 정합성 점검 결과

### 일치하는 부분
- 단일 학습 스크립트 내 stage 분리(`train_jepa`, `train_decoder`) 구조 존재.
- FSQ 인덱스 패킹/토큰 통계 계산 함수가 포함되어 토큰/sec 측정 로직이 있음.
- stage2에서 encoder freezing 및 decoder/adversarial 학습 경로가 있음.
- DeepSpeed 체크포인트를 일반 `.pt`로 합치는 보조 스크립트 제공.

### 불명확/부족한 부분
- 저장소 파일 수가 적고(README + 2 scripts) 실험 재현에 필요한 설정 파일(`ds_config.json`)과 데이터 준비 파이프라인이 저장소 내부에 완결되어 있지 않음.
- 실행 예시에서 파일명 불일치가 관찰됨(README 예시는 `train_jepa_fsqvae_hifigan.py`, 실제 파일은 `train_fsqvae_jepa.py`).
- 논문의 정량 지표를 재현하기 위한 하이퍼파라미터/데이터셋 상세가 코드만으로 충분히 고정되지 않아, “즉시 재현”보다는 “학습 코드 공개”에 가까운 상태.
- 웹에서 조회된 raw 스크립트는 단일 라인 형태로 서빙되어 가독성과 코드 감사성이 떨어짐.

## 본 저장소에서의 보강 방향
- 학습/추론 최소 재현 가능한 **독립 실행형 파이프라인** 제공.
- DAAM 개념을 반영한 간결한 모듈화 구현 (`DAAM`, `FSQ`, stage1/stage2 trainer).
- README에 데이터 포맷, 단계별 명령, 실패 대응 팁, 산출물 설명까지 포함.

## 결론
- 공개 코드는 논문의 큰 구조(2-stage + DAAM/FSQ + decoder)를 반영하고 있으나,
  즉시 재현 가능한 형태로는 문서/구성 요소가 일부 부족하다.
- 따라서 본 저장소에서는 **재현성 중심의 baseline 구현 + 상세 문서화**를 추가하였다.
