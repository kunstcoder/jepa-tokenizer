import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/train.jsonl')
    ap.add_argument('--count', type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for i in range(args.count):
            f.write(json.dumps({'path': '', 'id': i}) + '\n')


if __name__ == '__main__':
    main()
