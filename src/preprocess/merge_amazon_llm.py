# -*- coding: utf-8 -*-
import argparse
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genuine", required=True, help="genuine reviews csv")
    ap.add_argument("--spam", required=True, help="LLM spam reviews csv")
    ap.add_argument("--out", required=True, help="output csv")
    args = ap.parse_args()

    df_genuine = pd.read_csv(args.genuine)
    df_genuine = df_genuine.copy()
    df_genuine["label"] = 0

    df_spam = pd.read_csv(args.spam)
    df_spam = df_spam.copy()
    df_spam["label"] = 1

    df = pd.concat([df_genuine, df_spam], axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote: {args.out}")
    print("rows:", len(df), "spam_rate:", float(df["label"].mean()))


if __name__ == "__main__":
    main()
