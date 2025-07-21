import pandas as pd
import re

# Load CSV
csv_path = r"File Path"
df = pd.read_csv(csv_path)

# --- Step 1: Extract Original Message Body (If Forwarded/Replied) ---
def extract_original_body(body):
    body = str(body)
    while True:
        match = re.search(
            r"(--- Forwarded by .+? on .+? ---|-----Original Message-----)\s*"
            r"From:\s+.+?\n"
            r"To:\s+.+?\n"
            r"Date:\s+.+?\n"
            r"Subject:\s+.+?\n\n"
            r"(?P<body>.*)$",
            body,
            re.DOTALL
        )
        if match:
            body = match.group("body")
        else:
            break
    return body

# --- Step 2: Basic Body Cleaning ---
def clean_body(text):
    text = str(text)
    text = re.sub(r"(?m)^[>].*", "", text)  # Remove quoted lines
    text = re.sub(r"(?i)--+.*", "", text)   # Remove signatures
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"\S+@\S+", "", text)     # Emails
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Non-letters
    text = re.sub(r"\s+", " ", text)        # Extra whitespace
    return text.strip().lower()

# --- Step 3: Apply Extraction + Cleaning In-Place ---
df['body_chunk'] = df['body_chunk'].apply(lambda x: clean_body(extract_original_body(x)))

# --- Step 4: Save Cleaned File ---
output_csv = r"E:\SchmalkaldenAdventure\Sem 2\HMI\Enron\parsed_chunked_emails_cleaned.csv"
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Cleaned and saved to: {output_csv}")
