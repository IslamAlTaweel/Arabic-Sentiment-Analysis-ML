import pandas as pd

labels = {"POS", "NEG", "OBJ", "NEUTRAL"}
data = []

with open("Arabic-Tweets_Dataset.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Last token must be a label
        if parts[-1] in labels:
            label = parts[-1]
            text = " ".join(parts[:-1])
            data.append([text, label])
        else:
            # skip malformed lines safely
            print("Skipped:", line)

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("Arabic-Tweets_Dataset.csv", index=False, encoding="utf-8-sig")
