"""
Generates data/raw/sample_twcs.csv -- a small, HAND-CONSTRUCTED dataset in the
exact schema of the real Kaggle "Customer Support on Twitter" dump
(tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id), for brand AmazonHelp.

WHY THIS EXISTS: the environment used to build this repo has no internet
access, so the real ~3M-row Kaggle CSV can't be downloaded here. This script
fabricates ~180 realistic customer/agent tweet pairs (patterned after common
structures in the real dataset and Amazon's actual support account style) so
the pipeline is runnable end-to-end immediately, without waiting on a
multi-GB download.

BEFORE YOU SUBMIT: download the real file from Kaggle
(thoughtvector/customer-support-on-twitter), place it at data/raw/twcs.csv,
and run the pipeline with --data data/raw/twcs.csv instead of the sample.
The code path is identical either way -- only the input file changes.
Do NOT submit results computed only on this synthetic sample; it exists to
prove the pipeline runs, not to stand in for the real evaluation.
"""
import csv
import random

random.seed(7)

AUTHOR_POOL = [f"cust_{i:04d}" for i in range(1, 140)]

TEMPLATES = [
    ("order_status_delay",
     "@AmazonHelp my order #{oid} was supposed to arrive yesterday and tracking hasn't updated in 3 days. where is it??",
     "@{user} I'm sorry for the delay! I've located order #{oid} -- please DM us your order confirmation email so I can check the carrier update for you. ^KL"),
    ("order_status_delay",
     "@AmazonHelp still no sign of my package, it's been 6 days past the delivery date. this is ridiculous",
     "@{user} That's not the experience we want for you. Can you send us your order number via DM so we can look into the carrier delay right away? ^MJ"),
    ("order_status_delay",
     "@AmazonHelp order #{oid} says 'out for delivery' for the 4th day in a row now, what's going on",
     "@{user} Sorry for the runaround! Please DM order #{oid} and we'll check directly with the carrier for you. ^AR"),
    ("order_status_delay",
     "@AmazonHelp tracking shows my package delivered but I never got anything, checked with neighbors too",
     "@{user} That's concerning, sorry! Please DM your order number and we'll open a missing-package investigation with the carrier. ^KL"),
    ("order_status_delay",
     "@AmazonHelp why does my delivery estimate keep moving further out every single day",
     "@{user} Apologies for the shifting estimate! DM your order number and I'll pull the latest carrier status for you. ^MJ"),
    ("damaged_or_wrong_item",
     "@AmazonHelp received my order today and the item is completely smashed, box was soaking wet too",
     "@{user} So sorry to see this! Please DM us photos of the damaged item and your order number and we'll get a replacement or refund started. ^AR"),
    ("damaged_or_wrong_item",
     "@AmazonHelp you sent me the wrong color headphones, I ordered black and got white",
     "@{user} Apologies for the mix-up! Send us your order ID in a DM and we'll arrange a free return + the correct item shipped out. ^KL"),
    ("damaged_or_wrong_item",
     "@AmazonHelp order #{oid} arrived missing half the parts, box wasn't even sealed properly",
     "@{user} Sorry about that! DM your order number #{oid} with photos and we'll get replacement parts or a full replacement sent. ^MJ"),
    ("damaged_or_wrong_item",
     "@AmazonHelp the blender I ordered arrived with a cracked jar, clearly used not new",
     "@{user} That's not okay, sorry! Please DM your order number and a photo so we can process a replacement right away. ^AR"),
    ("refund_or_return",
     "@AmazonHelp I returned my order 2 weeks ago and still haven't gotten my refund, this is unacceptable",
     "@{user} I understand the frustration. Could you DM your order number and return tracking so I can check the refund status for you? ^MJ"),
    ("refund_or_return",
     "@AmazonHelp how do I get a refund for an order I never received",
     "@{user} We can help with that. Please DM us the order number and we'll start a refund if the carrier confirms it wasn't delivered. ^AR"),
    ("refund_or_return",
     "@AmazonHelp return window says closed but the item just arrived broken, what do I do now",
     "@{user} Sorry for the trouble! Send us a DM with the order number and photos and we can still process this as a damaged-item exception. ^KL"),
    ("refund_or_return",
     "@AmazonHelp requested a refund 10 days ago for order #{oid}, still shows 'processing'",
     "@{user} Apologies for the wait on #{oid} -- DM us and I'll check directly with the refunds team on the status. ^MJ"),
    ("billing_charge_dispute",
     "@AmazonHelp I was charged twice for the same order, $89.99 showed up on my card two times",
     "@{user} Sorry about that! Please DM your order number and the last 4 digits of the card charged so we can investigate the duplicate charge. ^KL"),
    ("billing_charge_dispute",
     "@AmazonHelp I cancelled my Prime membership last month but you just charged me again",
     "@{user} Apologies for the confusion. Send us a DM with the email on the account and we'll review the cancellation and refund if applicable. ^MJ"),
    ("billing_charge_dispute",
     "@AmazonHelp charged $34.50 for an order that got cancelled automatically, why wasn't I refunded",
     "@{user} Sorry for the mix-up! DM the order number and we'll check why the refund didn't process automatically. ^AR"),
    ("billing_charge_dispute",
     "@AmazonHelp my card was charged $212 but I only bought a $40 item, please explain",
     "@{user} That doesn't look right, sorry! Please DM your order number and the charge amount so we can investigate immediately. ^KL"),
    ("account_access_issue",
     "@AmazonHelp I can't log into my account, it says my password is wrong but I know it's right, did I get hacked?",
     "@{user} Let's get this sorted securely -- please DM us so we can verify your identity and help restore access to your account. ^AR"),
    ("account_access_issue",
     "@AmazonHelp getting a 2FA code loop, can't get past login on my new phone",
     "@{user} Sorry for the trouble! DM us the email on file and we'll help you regain access step by step. ^KL"),
    ("account_access_issue",
     "@AmazonHelp someone placed an order on my account that I didn't make, is my account compromised",
     "@{user} Please DM us right away -- we'll help secure your account and look into the unauthorized order. ^MJ"),
    ("account_access_issue",
     "@AmazonHelp locked out of my account after too many failed login attempts, need to get back in urgently",
     "@{user} Sorry for the lockout! DM us your account email and we'll walk you through regaining access. ^AR"),
    ("cancellation_request",
     "@AmazonHelp need to cancel order #{oid} before it ships, ordered the wrong size",
     "@{user} No problem, let's try to catch it in time! DM us the order number now and we'll check if it can still be cancelled. ^MJ"),
    ("cancellation_request",
     "@AmazonHelp how do I cancel my Prime free trial before it charges me",
     "@{user} You can cancel anytime from Your Account > Prime Membership > End Membership. Happy to help if you hit any issues -- just DM us! ^AR"),
    ("cancellation_request",
     "@AmazonHelp accidentally ordered 2 of the same item, need to cancel one of them fast",
     "@{user} Let's see what we can do! DM us both order numbers and we'll try to cancel the duplicate before it ships. ^KL"),
    ("cancellation_request",
     "@AmazonHelp trying to cancel a subscribe & save order but the app won't let me",
     "@{user} Sorry for the app trouble! DM us the order number and we'll cancel it manually on our end. ^MJ"),
    ("product_or_service_question",
     "@AmazonHelp does the Echo Dot work without wifi at all or is it useless offline?",
     "@{user} Great question! Echo Dot needs wifi for most features like Alexa requests, though a few local smart home controls may still work. ^KL"),
    ("product_or_service_question",
     "@AmazonHelp is prime video included free with a regular prime membership or extra cost",
     "@{user} Prime Video is included at no extra cost with your Prime membership! Enjoy 🎬 ^MJ"),
    ("product_or_service_question",
     "@AmazonHelp can I share my prime benefits with a family member living at a different address",
     "@{user} Yes! Amazon Household lets you share select Prime benefits with another adult, even at a different address. ^AR"),
    ("product_or_service_question",
     "@AmazonHelp does this kindle model support audiobooks or just ebooks",
     "@{user} Good question! That Kindle model supports Audible audiobooks via Bluetooth in addition to ebooks. ^KL"),
    ("other_complaint",
     "@AmazonHelp the delivery driver just threw my package over the fence and it landed in the pool, unbelievable",
     "@{user} I'm really sorry to hear that. Please DM us your order number and details so we can look into this and make it right. ^AR"),
    ("other_complaint",
     "@AmazonHelp app keeps crashing every time I try to check out on iOS, anyone else having this issue",
     "@{user} Sorry for the inconvenience! Can you DM us your device model and app version so our team can look into this? ^KL"),
    ("other_complaint",
     "@AmazonHelp delivery photo shows package at a door that isn't even mine, wrong address entirely",
     "@{user} Sorry about that mix-up! Please DM your order number so we can locate your package and get it redelivered. ^MJ"),
    ("other_complaint",
     "@AmazonHelp packaging was so excessive for one small item, a tiny USB cable in a huge box with tons of air pillows",
     "@{user} Thanks for the feedback, we're always working to improve packaging. I'll pass this along to the team! ^AR"),
]

ANGRY_PREFIXES = ["", "", "", "this is the third time this has happened. ", "absolutely done with this. ", "worst experience ever, "]

rows = []
tid = 100000


def next_id():
    global tid
    tid += 1
    return tid


ground_truth = []  # tweet_id -> intent, kept separately: the real Kaggle CSV
                    # has no intent column either, so this mirrors the fact
                    # that intent labels always have to come from a human
                    # labeling pass (build_golden_set.py does that pass).

for i in range(700):
    intent, cust_t, agent_t = random.choice(TEMPLATES)
    user = random.choice(AUTHOR_POOL)
    oid = random.randint(100000000, 999999999)
    prefix = random.choice(ANGRY_PREFIXES)
    cust_text = (prefix + cust_t).format(oid=oid, user=user)
    agent_text = agent_t.format(oid=oid, user=user)

    cust_id = next_id()
    agent_id = next_id()
    ground_truth.append({"tweet_id": cust_id, "intent": intent, "is_angry_prefix": prefix != ""})

    rows.append({
        "tweet_id": cust_id,
        "author_id": user,
        "inbound": "True",
        "created_at": f"Wed Jan {1 + (i % 28):02d} 12:{i % 60:02d}:00 +0000 2026",
        "text": cust_text,
        "response_tweet_id": str(agent_id),
        "in_response_to_tweet_id": "",
    })
    rows.append({
        "tweet_id": agent_id,
        "author_id": "AmazonHelp",
        "inbound": "False",
        "created_at": f"Wed Jan {1 + (i % 28):02d} 12:{(i + 3) % 60:02d}:00 +0000 2026",
        "text": agent_text,
        "response_tweet_id": "",
        "in_response_to_tweet_id": str(cust_id),
    })

random.shuffle(rows)  # real dump isn't neatly paired/ordered either

with open("data/raw/sample_twcs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "tweet_id", "author_id", "inbound", "created_at", "text",
        "response_tweet_id", "in_response_to_tweet_id",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to data/raw/sample_twcs.csv")

with open("data/raw/sample_ground_truth_intents.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tweet_id", "intent", "is_angry_prefix"])
    writer.writeheader()
    writer.writerows(ground_truth)

print(f"Wrote {len(ground_truth)} rows to data/raw/sample_ground_truth_intents.csv "
      f"(NOTE: this file is a stand-in for a human labeling pass -- it exists "
      f"only because this sample was template-generated. It is used by "
      f"build_golden_set.py to simulate the labeling step honestly. On the "
      f"real Kaggle data there is no such file and eval/build_golden_set.py's "
      f"sampling logic is meant to be paired with actual manual labeling.)")
