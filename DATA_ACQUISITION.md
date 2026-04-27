# 논문 데이터 확보 및 검증 가이드 (LibriLight)

## 1) 논문 기준 데이터셋 확인
- JEPA Tokenizer 논문(arXiv:2512.07168) 4.1절은 학습 데이터로 **LibriLight**를 명시합니다.
- 논문에 기재된 설정:
  - Training split: 약 9,000시간
  - Sample rate: 24kHz
  - Max audio length: 15초

## 2) 공식 확보 경로
공식 데이터 준비 절차는 `facebookresearch/libri-light` 저장소의 `data_preparation/README.md`에 정리되어 있습니다.

해당 문서에는 다음이 포함됩니다.
- LibriLight unlabelled subset 다운로드 링크
  - `small.tar` (577h)
  - `medium.tar` (5193h)
  - `large.tar` (51934h)
- 각 tar의 MD5
- limited supervision 세트(`librispeech_finetuning.tgz`)
- 평가용 LibriSpeech(dev/test) 다운로드 경로(OpenSLR)

## 3) 권장 확보 절차
### 3.1 대용량 무라벨 음성 확보
```bash
mkdir -p data/raw/librilight && cd data/raw/librilight

# 필요 용량에 맞춰 선택 다운로드
wget https://dl.fbaipublicfiles.com/librilight/data/small.tar
wget https://dl.fbaipublicfiles.com/librilight/data/medium.tar
# wget https://dl.fbaipublicfiles.com/librilight/data/large.tar
```

### 3.2 무결성 검증(MD5)
```bash
# 공식 md5 (facebookresearch/libri-light data_preparation 문서 기준)
echo "c49207eb86a8e8ac895561c37232041e  small.tar" | md5sum -c -
echo "c75e7ac62471bfbf2db77528d62a9b74  medium.tar" | md5sum -c -
echo "4dfbac018f50b99797ece101fc9f0c30  large.tar" | md5sum -c -
```

### 3.3 압축 해제
```bash
mkdir -p extracted
# 예시: small/medium만 사용하는 경우
tar -xf small.tar -C extracted
tar -xf medium.tar -C extracted
```

### 3.4 본 저장소 학습 포맷(JSONL) 생성
이 저장소는 한 줄당 `{"path": "..."}` 형태 JSONL을 사용합니다.

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('data/raw/librilight/extracted')
# flac 기준. 필요시 wav/mp3 확장
audio_files = sorted(root.rglob('*.flac'))
out = Path('data/train.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    for p in audio_files:
        f.write(json.dumps({'path': str(p)}) + '\n')
print(f'wrote {len(audio_files)} lines to {out}')
PY
```

### 3.5 샘플레이트/길이 확인(논문 설정 정합)
```bash
python - <<'PY'
import json, random
import torchaudio

with open('data/train.jsonl', 'r', encoding='utf-8') as f:
    items = [json.loads(x)['path'] for x in f if x.strip()]

samples = random.sample(items, min(20, len(items)))
errs = 0
for p in samples:
    info = torchaudio.info(p)
    if info.sample_rate != 24000:
        errs += 1
print(f'checked={len(samples)}, non_24k={errs}')
print('참고: 학습 코드가 필요 시 24k로 리샘플링합니다.')
PY
```

## 4) 본 저장소 코드와의 연결
- `src/jepa_tokenizer.py`는 JSONL의 `path` 필드를 읽어 오디오를 로드합니다.
- 오디오 로드 실패 시 스모크 테스트를 위해 랜덤 노이즈 fallback이 있습니다.
- 따라서 실제 재현에서는 반드시 올바른 데이터 경로/오디오 파일 존재를 확인해야 합니다.

## 5) 이번 턴의 검증 결과
- 논문 본문(4.1 Dataset)과 공개 코드 문서를 교차 검토해, **LibriLight 확보 경로와 학습 포맷 변환 절차를 확정**했습니다.
- 다만 이 실행 환경은 외부 대용량 파일 HEAD/다운로드 요청이 403으로 차단되어 실제 파일 무결성 검증(MD5)까지는 수행하지 못했습니다.
