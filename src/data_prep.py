"""
Loads the raw Twitter customer-support CSV (real Kaggle file or the synthetic
sample -- same schema), filters to one brand, and reconstructs
(customer_message -> brand_reply) pairs using the response_tweet_id /
in_response_to_tweet_id links.

Real-dataset gotchas this handles:
- inbound is a string "True"/"False" in the raw CSV, not a bool.
- response_tweet_id can contain multiple comma-separated ids (a tweet can be
  "responded to" by more than one downstream tweet); we only take the first
  and drop the pair if it doesn't lead to the brand account.
- A meaningful fraction of customer tweets never got a reply at all (churned
  threads, or the reply is to a different brand tweet not in our slice). We
  keep customer-only rows too (usable for classification) but only usable for
  "grounded reply" retrieval when a real brand reply exists.
- Tweets are full of @mentions, RT noise, and multi-tweet threads from the
  same customer. We do light cleaning, not aggressive normalization, since
  over-cleaning destroys the signal we need for intent classification.
"""
from __future__ import annotations

import re
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["inbound"] = df["inbound"].map({"True": True, "False": False, "true": True, "false": False})
    return df


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_brand_pairs(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """Returns one row per customer message directed at `brand_handle`,
    joined with the brand's reply text when one exists.

    Columns: customer_tweet_id, customer_text, customer_text_clean,
             author_id, brand_reply_id, brand_reply_text (nullable)
    """
    brand_handle = brand_handle.lstrip("@")

    is_to_brand = df["inbound"] & df["text"].str.contains(f"@{brand_handle}", case=False, na=False)
    customer_msgs = df[is_to_brand].copy()

    brand_msgs = df[(~df["inbound"]) & (df["author_id"].str.lower() == brand_handle.lower())].copy()
    brand_msgs = brand_msgs.set_index("tweet_id")

    def first_response_id(cell: str):
        if not cell:
            return None
        return cell.split(",")[0].strip()

    customer_msgs["first_response_id"] = customer_msgs["response_tweet_id"].map(first_response_id)

    def lookup_reply(rid):
        if rid is None or rid == "" or rid not in brand_msgs.index:
            return None
        return brand_msgs.loc[rid, "text"]

    customer_msgs["brand_reply_text"] = customer_msgs["first_response_id"].map(lookup_reply)
    customer_msgs["customer_text_clean"] = customer_msgs["text"].map(clean_text)
    customer_msgs["brand_reply_text_clean"] = customer_msgs["brand_reply_text"].map(
        lambda t: clean_text(t) if isinstance(t, str) else None
    )

    out = customer_msgs.rename(columns={"tweet_id": "customer_tweet_id"})[
        ["customer_tweet_id", "author_id", "text", "customer_text_clean",
         "first_response_id", "brand_reply_text", "brand_reply_text_clean"]
    ].reset_index(drop=True)

    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_twcs.csv"
    brand = sys.argv[2] if len(sys.argv) > 2 else "AmazonHelp"
    df = load_raw(path)
    pairs = build_brand_pairs(df, brand)
    print(f"Loaded {len(df)} raw rows, {len(pairs)} customer->{brand} messages, "
          f"{pairs['brand_reply_text'].notna().sum()} with a matched brand reply.")
    print(pairs.head(3).to_string())
