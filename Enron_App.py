import os
import email
from email import policy
import pandas as pd
import re
from datetime import datetime
from collections import Counter

# Step 1: Define a function to parse a single email file
def parse_email(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        msg = email.message_from_file(file, policy=policy.default)

        # Extract fields
        from_ = msg.get('From', '')
        to_ = msg.get('To', '')
        subject = msg.get('Subject', '')
        date = msg.get('Date', '')
        body = ""

        # Step 2: Get the body of the email
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors='ignore')
        else:
            body = msg.get_payload(decode=True).decode(errors='ignore')

        # Step 3: Normalize and clean fields
        body = clean_email_body(body)
        from_ = from_.lower() if from_ else ''
        to_ = to_.lower() if to_ else ''
        subject = subject.strip() if subject else ''

        # Step 4: Convert date string to datetime object
        try:
            date_obj = email.utils.parsedate_to_datetime(date)
        except:
            date_obj = None

        # Step 5: Calculate word count
        word_count = len(body.split())

        return {
            'from': from_,
            'to': to_,
            'subject': subject,
            'date': date_obj,
            'body': body,
            'word_count': word_count
        }

# Step 6: Clean the body text of the email
def clean_email_body(body):
    if not isinstance(body, str):
        return ""
    body = re.split(r'(-{2,}Original Message-{2,}|On .* wrote:)', body)[0]
    body = re.split(r'(--\s*$|Thanks,|Regards,|Best,)', body)[0]
    body = re.sub(r'[^a-zA-Z0-9\s.,?!]', ' ', body)
    body = re.sub(r'\s+', ' ', body)
    return body.lower().strip()

# Step 7: Walk through all files and try parsing them
def process_email_directory(directory_path):
    all_emails = []
    total_files_scanned = 0
    successfully_parsed = 0
    failed_files = []

    for root, _, files in os.walk(directory_path):
        for filename in files:
            total_files_scanned += 1
            file_path = os.path.join(root, filename)
            try:
                parsed_email = parse_email(file_path)
                all_emails.append(parsed_email)
                successfully_parsed += 1
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"[Not Ok] Error parsing {file_path}: {e}")

    if failed_files:
        with open("parsing_errors.log", "w", encoding="utf-8") as f:
            for path, error in failed_files:
                f.write(f"{path}: {error}\n")

    df = pd.DataFrame(all_emails)

    print("\n[Summary]")
    print(f" Total files scanned: {total_files_scanned}")
    print(f" Emails successfully processed: {successfully_parsed}")
    print(f" Failed to process: {len(failed_files)}")
    print(f" Unique senders: {df['from'].nunique()}")
    print(f"Replies/Forwards: {df['subject'].str.contains('re:|fw:', case=False, na=False).sum()}")
    print(" Most common subject lines:")
    print(df['subject'].value_counts().head(5))
    print(" Average word count:", df['word_count'].mean())
    print("\n[ok] All files done.\n")

    return df

# Step 8: Save the processed data to CSV
def save_cleaned_data(df, output_path="cleaned_data/enron_cleaned.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[ok] Saved cleaned data to {output_path}")

# Main pipeline
if __name__ == "__main__":
    enron_data_path = r"E:\SchmalkaldenAdventure\Sem 2\HMI\Enron\INEnron"
    df = process_email_directory(enron_data_path)
    print(f"[Ok] Total emails processed: {len(df)}")
    save_cleaned_data(df, "cleaned_data/enron_cleaned.csv")
