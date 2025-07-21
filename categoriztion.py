import pandas as pd

# === Load your CSV with combined text ===
input_csv = r"File Path"
df = pd.read_csv(input_csv)

# === Define the categorization function ===
def rule_based_category(text):
    text = str(text).lower()
    
    finance_keywords = ['budget', 'invoice', 'payment', 'expense', 'cost']
    hr_keywords = ['recruitment', 'staff', 'benefits', 'resignation', 'hiring']
    legal_keywords = ['contract', 'compliance', 'lawsuit', 'regulation', 'legal']
    meetings_keywords = ['agenda', 'meeting', 'minutes', 'schedule', 'conference']
    technical_keywords = ['server', 'system', 'software', 'bug', 'issue']

    if any(word in text for word in finance_keywords):
        return 'Finance'
    elif any(word in text for word in hr_keywords):
        return 'HR'
    elif any(word in text for word in legal_keywords):
        return 'Legal'
    elif any(word in text for word in meetings_keywords):
        return 'Meetings'
    elif any(word in text for word in technical_keywords):
        return 'Technical'
    else:
        return 'Other'

# === Apply categorization ===
df['rule_category'] = df['text'].apply(rule_based_category)

# === Save the categorized dataset ===
output_csv = r"File Path"
df.to_csv(output_csv, index=False)
print(f"Rule-based categorization applied and saved to: {output_csv}")

# === Optional: Display sample results ===
print(df[['subject', 'rule_category']].head(10))
