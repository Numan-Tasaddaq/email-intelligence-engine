import pandas as pd
from dateutil import parser

# === Step 0: Load the dataset ===
input_csv = r"File Path"
df = pd.read_csv(input_csv)

# Fill missing text fields to avoid errors
df[['subject', 'body_chunk', 'from', 'to', 'date']] = df[['subject', 'body_chunk', 'from', 'to', 'date']].fillna('')

# === Step 1: Aggregate chunks into full email bodies ===
# Sort by chunk_id to keep order correct, then join body chunks
df = df.sort_values(by=['file', 'chunk_id'])
full_bodies = df.groupby('file').agg({
    'subject': 'first',       # assume subject is same across chunks
    'from': 'first',
    'to': 'first',
    'date': 'first',          # take date from first chunk
    'body_chunk': lambda x: ' '.join(x.astype(str))  # concatenate all chunks
}).reset_index()

# === Step 1.5: Parse date string into datetime object using dateutil ===
def safe_parse_date(d):
    try:
        return parser.parse(d)
    except Exception:
        return pd.NaT

full_bodies['parsed_date'] = full_bodies['date'].apply(safe_parse_date)

# === Step 2: Define classification functions ===
def classify_topic(subject, body):
    text = (str(subject) + " " + str(body)).lower()
    if any(word in text for word in ["budget", "financial", "forecast", "cost", "revenue", "q1", "q2"]):
        return "Finance"
    elif any(word in text for word in ["legal", "law", "subpoena", "regulatory", "ferc", "compliance"]):
        return "Legal"
    elif any(word in text for word in ["hiring", "hr", "employee", "job", "recommendation", "resume"]):
        return "HR"
    elif any(word in text for word in ["meeting", "schedule", "desk", "procedure", "logistics"]):
        return "Operations"
    elif any(word in text for word in ["merger", "strategy", "plan", "integration", "partnership"]):
        return "Strategy"
    elif any(word in text for word in ["technical", "interconnection", "system", "configuration", "policy"]):
        return "Technical"
    else:
        return "General/Other"

def classify_urgency(body):
    text = str(body).lower()
    if any(word in text for word in ["please", "can you", "do you", "need you to", "would you"]):
        return "Request / Task"
    elif any(word in text for word in ["just a reminder", "as discussed earlier", "follow up", "note that"]):
        return "Reminder"
    elif any(word in text for word in ["urgent", "issue", "problem", "concern", "asap", "immediate"]):
        return "Escalation"
    else:
        return "Informational"

def classify_role(email):
    email = str(email).lower()
    if any(name in email for name in ["kenneth.lay", "jeff.skilling", "steven.kean"]):
        return "Executive"
    elif any(word in email for word in ["legal", "attorney", "counsel"]):
        return "Legal"
    elif "@enron.com" in email:
        return "Employee"
    else:
        return "External"

# === Step 3: Apply classifications on full emails ===
full_bodies['topic_label'] = full_bodies.apply(lambda row: classify_topic(row['subject'], row['body_chunk']), axis=1)
full_bodies['urgency_label'] = full_bodies['body_chunk'].apply(classify_urgency)
full_bodies['role_from'] = full_bodies['from'].apply(classify_role)
full_bodies['role_to'] = full_bodies['to'].apply(classify_role)

# === Step 4: Save results ===
output_csv = r"File Path"
full_bodies.to_csv(output_csv, index=False)

print(f"Full email classification complete. Results saved to: {output_csv}")
