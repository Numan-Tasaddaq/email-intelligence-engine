import os
from email import message_from_string
import pandas as pd

# ========== STEP 1: Traverse and Load Emails ==========
def load_emails(root_dir):
    email_data = []
    for user_dir, _, files in os.walk(root_dir):
        for filename in files:
            filepath = os.path.join(user_dir, filename)
            try:
                with open(filepath, 'r', encoding='latin1') as f:
                    raw_email = f.read()
                    email_data.append((filepath, raw_email))
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")
    return email_data

# ========== STEP 2: Parse Emails ==========
def parse_email(raw_email):
    msg = message_from_string(raw_email)
    email_content = {
        "from": msg.get("From"),
        "to": msg.get("To"),
        "date": msg.get("Date"),
        "subject": msg.get("Subject"),
        "body": get_body(msg),
    }
    return email_content

def get_body(msg):
    if msg.is_multipart():
        parts = [part.get_payload(decode=True) 
                 for part in msg.get_payload() 
                 if part.get_content_type() == 'text/plain']
        return ''.join([p.decode(errors='ignore') if p else '' for p in parts])
    else:
        payload = msg.get_payload(decode=True)
        return payload.decode(errors='ignore') if payload else ''

# ========== STEP 3: Chunking ==========
def chunk_text(text, max_tokens=500):
    words = text.split()
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

# ========== STEP 4: Main Processing ==========
def process_directory(root_dir):
    emails = load_emails(root_dir)
    parsed_chunks = []

    for filepath, raw_email in emails:
        parsed = parse_email(raw_email)
        chunks = chunk_text(parsed['body'])
        for i, chunk in enumerate(chunks):
            parsed_chunks.append({
                "file": filepath,
                "chunk_id": i,
                "from": parsed["from"],
                "to": parsed["to"],
                "date": parsed["date"],
                "subject": parsed["subject"],
                "body_chunk": chunk
            })

    return parsed_chunks

# ========== STEP 5: Save to CSV ==========
def save_chunks_to_csv(chunks, output_file):
    df = pd.DataFrame(chunks)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved {len(chunks)} chunks to {output_file}")

# ========== RUN ==========
if __name__ == "__main__":
    root_dir = r"File Path"
    output_csv = "enron_email_chunks.csv"

    chunks = process_directory(root_dir)
    save_chunks_to_csv(chunks, output_csv)
