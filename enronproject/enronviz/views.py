import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import Http404
from urllib.parse import unquote
import seaborn as sns
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from dateutil import parser
import google.generativeai as genai
import os
from concurrent.futures import ThreadPoolExecutor
from django.core.cache import cache
from django.conf import settings
from matplotlib.patches import Patch
# Configure Gemini API - Load key from Django settings
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
# Load data once
df = pd.read_csv("E:/SchmalkaldenAdventure/Sem 2/HMI/Enron/enron_rule_categorized.csv")

# === Preprocessing ===
df = df.dropna(subset=['body_chunk', 'rule_category', 'from', 'to', 'date'])
df['body_chunk'] = df['body_chunk'].astype(str)
df['rule_category'] = df['rule_category'].astype(str)
df['from'] = df['from'].str.strip().str.lower()
df['to'] = df['to'].str.replace(r'[\[\]\'\"]', '', regex=True)
df['to_list'] = df['to'].str.split(',')

# === Parse Dates ===
from dateutil import parser
def safe_parse_date(d):
    try:
        return parser.parse(d)
    except:
        return pd.NaT

df['parsed_date'] = df['date'].apply(safe_parse_date)
df = df.dropna(subset=['parsed_date'])
df['parsed_date'] = pd.to_datetime(df['parsed_date'], utc=True).dt.tz_convert(None)
df['year_month'] = df['parsed_date'].dt.to_period('M').astype(str)
df['year_quarter'] = df['parsed_date'].dt.to_period('Q').astype(str)



def home(request):
    return render(request, 'enronviz/home.html')

def story(request):
    return render(request, 'enronviz/story.html')

def agent(request):
    return render(request, 'enronviz/agent.html')

def visualization_list(request):

    return render(request, 'visualizations/list.html')

# views.py - Keep only the relevant parts

def dashboard(request):
    """Main dashboard view that serves the HTML template"""
    return render(request, 'viz/dashboard.html')

def get_visualization(request, viz_type):
    """AJAX endpoint that returns visualization as JSON with period filtering"""
    try:
        # Common period definitions
        period_names = {
            'before': 'Pre-Crisis (Jan 1999 - Jul 2001)',
            'during': 'Crisis Period (Aug - Dec 2001)',
            'after': 'Post-Crisis (Dec 2001 - Dec 2002)'
        }

        if viz_type == 'email-traffic':
            img, title, description = generate_email_traffic_plot()
            return JsonResponse({
                'image': img,
                'title': title,
                'description': description,
                'key_insights': [
                    "Email volume peaked in mid-2001 before the scandal broke",
                    "Notice the drop in communications after bankruptcy filing",
                    "Spikes may indicate crisis communication periods"
                ]
            })

        elif viz_type == 'active-employees':
            period = request.GET.get('period', 'during')
            img, title, description = generate_active_employees_plot(period)
            
            return JsonResponse({
                'image': img,
                'title': title,
                'description': description,
                'key_insights': [
                    f"Analysis of {period_names.get(period, 'selected period')}",
                    "Executives dominate the communication volume",
                    "Some employees show unusually high email activity",
                    "Key players in the scandal are clearly visible",
                    f"Notice how activity patterns differ from other periods"
                ]
            })

        elif viz_type == 'email-network':
            period = request.GET.get('period', 'during')
            img, title, description = generate_email_network_plot(period)
            
            return JsonResponse({
                'image': img,
                'title': title,
                'description': description,
                'key_insights': [
                    f"Network analysis for {period_names.get(period, 'selected period')}",
                    "Node size represents email volume",
                    "Edge thickness shows communication frequency",
                    "Red nodes indicate key executives",
                    f"Observe network changes during {period_names.get(period, 'this period')}"
                ]
            })

        elif viz_type == 'keyword-cloud':
            period = request.GET.get('period', 'during')
            img, title, description = generate_keyword_cloud(period)  # Pass period to generator
            
            return JsonResponse({
                'image': img,
                'title': title,
                'description': description,
                'key_insights': [
                    f"Keyword analysis for {period_names.get(period, 'selected period')}",
                    "Financial terms dominate communications ('price', 'energy', 'contract')",
                    "Time-sensitive words indicate urgency ('immediately', 'today', 'meeting')",
                    "Negative terms appear frequently ('problem', 'issue', 'concern')",
                    f"Notice how terminology changes during {period_names.get(period, 'this period')}",
                    "The inset bar chart provides precise frequency comparisons"
                ]
            })

        else:
            return JsonResponse({'error': 'Invalid visualization type'}, status=400)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def generate_email_traffic_plot():
    """Generate an improved bar chart for Enron email traffic over time with clear annotations"""
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import matplotlib.ticker as ticker
    from io import BytesIO
    import base64

    plt.switch_backend('Agg')
    
    try:
        # === Prepare data ===
        plot_df = df.copy()
        plot_df['date'] = pd.to_datetime(plot_df['parsed_date'])
        
        # Focus on relevant time range
        plot_df = plot_df[(plot_df['date'] >= '1999-01-01') & (plot_df['date'] <= '2002-12-31')]
        
        # Monthly email counts
        counts = plot_df.resample('MS', on='date').size()
        
        # === Color-code important events ===
        colors = []
        for date in counts.index:
            if pd.to_datetime('2001-08-01') <= date <= pd.to_datetime('2001-08-31'):
                colors.append('orange')  # CEO resignation
            elif pd.to_datetime('2001-10-01') <= date <= pd.to_datetime('2001-10-31'):
                colors.append('red')     # $618M loss
            elif pd.to_datetime('2001-12-01') <= date <= pd.to_datetime('2001-12-31'):
                colors.append('green')   # Bankruptcy
            else:
                colors.append('steelblue')

        # === Plot ===
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(counts.index, counts.values, width=20, color=colors, alpha=0.85)

        # === Annotate major events ===
        events = {
            '2001-08-14': ('CEO Resignation', 'orange'),
            '2001-10-16': ('$618M Loss', 'red'),
            '2001-12-02': ('Bankruptcy', 'green')
        }

        y_max = counts.max()
        for date_str, (label, color) in events.items():
            event_date = pd.to_datetime(date_str)
            ax.axvline(x=event_date, color=color, linestyle='--', alpha=0.9, linewidth=2)
            ax.text(event_date, y_max * 0.9, label, rotation=90, va='top', ha='right',
                    bbox=dict(facecolor='white', edgecolor=color, alpha=0.8), fontsize=10)

        # === Label top 15 spikes ===
        top_n = 15
        top_indices = counts.sort_values(ascending=False).head(top_n).index
        for bar, date, value in zip(bars, counts.index, counts.values):
            if date in top_indices:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        str(value), ha='center', va='bottom', fontsize=9)

        # === Format axes ===
        ax.set_title(' Enron Email Traffic Over Time (1999–2002)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Emails Sent', fontsize=12)
        ax.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(12))

        ax.grid(True, linestyle=':', alpha=0.4)

        
        plt.tight_layout()

        # === Save image to base64 ===
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return img_base64, "Email Traffic Over Time", "Monthly email volume with key company events marked"

    except Exception as e:
        plt.close()
        raise Exception(f"Visualization error: {str(e)}")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from io import BytesIO
import base64
import textwrap
def generate_active_employees_plot(period='during'):
    """Generate a single horizontal bar chart of most active email senders with clear time periods."""
    plt.switch_backend('Agg')

    try:
        # Define executives to exclude
        executives = {
            'kenneth.lay@enron.com',
            'jeff.skilling@enron.com',
            'andrew.fastow@enron.com',
            'rebecca.mark@enron.com',
            'greg.whalley@enron.com'
        }

        # Date ranges with display names
        periods = {
            'before': ('1999-01-01', '2001-07-31', "Pre-Crisis (Jan 1999 - Jul 2001)"),
            'during': ('2001-08-01', '2001-12-02', "Crisis Period (Aug - Dec 2001)"),
            'after': ('2001-12-03', '2002-12-31', "Post-Crisis (Dec 2001 - Dec 2002)")
        }

        # Validate period input
        if period not in periods:
            period = 'during'  # default to 'during' if invalid period provided

        # Get the date range and display name for selected period
        start, end, period_title = periods[period]

        # Filter data and prepare top senders
        filtered_df = df[(df['parsed_date'] >= start) & (df['parsed_date'] <= end)]
        non_exec_df = filtered_df[~filtered_df['from'].isin(executives)]
        sender_counts = non_exec_df['from'].value_counts().head(15)
        
        # Prepare labels (just usernames without @enron.com)
        labels = [email.split('@')[0].replace('.', ' ').title() for email in sender_counts.index]

        # Create figure with large dimensions
        fig, ax = plt.subplots(figsize=(22, 12))
        
        # Create horizontal bars with period-specific color
        color = {
            'before': '#1f77b4',  # blue
            'during': '#d62728',  # red
            'after': '#2ca02c'     # green
        }[period]
        
        bars = ax.barh(labels, sender_counts.values, 
                      color=color, height=0.75, alpha=0.85)

        # Style the plot
        ax.set_title(f"Top 15 Most Active Employees\n{period_title}", 
                   fontsize=24, pad=25, fontweight='bold')
        ax.set_xlabel("Number of Emails Sent", fontsize=20, labelpad=15)
        ax.tick_params(axis='both', labelsize=18)
        ax.invert_yaxis()  # highest at top
        
        # Add value labels to bars
        max_val = sender_counts.values.max()
        ax.set_xlim(0, max_val * 1.2)  # add 20% buffer
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + max_val*0.03,  # position label slightly right of bar
                   bar.get_y() + bar.get_height()/2,
                   f'{int(width):,}',  # format with thousands separator
                   va='center', 
                   fontsize=18,
                   fontweight='bold')

        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        
        # Adjust layout
        plt.tight_layout()

        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return img_base64, f"Top Active Employees: {period_title}", "Showing top 15 email senders"

    except Exception as e:
        plt.close()
        raise Exception(f"Visualization error: {str(e)}")

def generate_email_network_plot(period='during'):
    """Generate a network graph of email communication during a specific Enron timeline phase."""
    plt.switch_backend('Agg')

    try:
        # Timeline periods
        periods = {
            'before': ('1999-01-01', '2001-07-31', "Pre-Crisis"),
            'during': ('2001-08-01', '2001-12-02', "Crisis Period"),
            'after': ('2001-12-03', '2002-12-31', "Post-Crisis")
        }

        # Default to 'during' if invalid
        if period not in periods:
            period = 'during'

        start_date, end_date, period_title = periods[period]

        # Filter dataset based on period
        period_df = df[(df['parsed_date'] >= start_date) & (df['parsed_date'] <= end_date)]

        # Limit data for performance
        sample_df = period_df.sample(n=5000, random_state=42) if len(period_df) > 5000 else period_df

        # Build edge list
        edge_list = []
        for _, row in sample_df.iterrows():
            sender = row['from']
            receivers = row['to_list']
            for receiver in receivers:
                receiver = receiver.strip().lower()
                if receiver:
                    edge_list.append((sender, receiver))

        # Create graph
        G = nx.Graph()
        edge_counts = defaultdict(int)
        for edge in edge_list:
            edge_counts[edge] += 1

        for edge, weight in edge_counts.items():
            G.add_edge(edge[0], edge[1], weight=weight)

        # Setup visual attributes
        executives_all = {
            'kenneth.lay@enron.com', 'jeff.skilling@enron.com',
            'andrew.fastow@enron.com', 'rebecca.mark@enron.com',
            'lou.pai@enron.com', 'greg.whalley@enron.com'
        }
        executives = executives_all & set(G.nodes())

        node_colors, node_sizes = [], []
        for node in G.nodes():
            if node in executives:
                node_colors.append('red')
                node_sizes.append(300)
            else:
                node_colors.append('skyblue')
                node_sizes.append(100)

        edge_widths = [G[u][v]['weight'] * 0.1 for u, v in G.edges()]
        pos = nx.spring_layout(G, k=0.15, iterations=20)

        # Draw network
        plt.figure(figsize=(16, 12))
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.3)

        # Label executives and highly connected nodes
        important_nodes = executives | {n for n in G.nodes() if G.degree(n) > 30}
        labels = {n: n.split('@')[0] for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')

        # Title & legend
        plt.title(f' Enron Email Communication Network – {period_title}', fontsize=18, pad=20)
        legend_elements = [
            Patch(facecolor='red', label='Executives'),
            Patch(facecolor='skyblue', label='Other Employees'),
            Patch(facecolor='none', label='Edge thickness = Email volume')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.tight_layout()

        # Encode image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return img_base64, f"Email Network: {period_title}", f"Email communications during {period_title.lower()}"

    except Exception as e:
        plt.close()
        raise Exception(f"Visualization error: {str(e)}")
from sklearn.feature_extraction.text import CountVectorizer

def generate_keyword_cloud(period='during'):
    """Generate a keyword-focused word cloud from Enron emails by period (no bar chart)."""
    plt.switch_backend('Agg')

    try:
        # === Define timeline periods ===
        periods = {
            'before': ('1999-01-01', '2001-07-31', "Pre-Crisis"),
            'during': ('2001-08-01', '2001-12-02', "Crisis Period"),
            'after': ('2001-12-03', '2002-12-31', "Post-Crisis")
        }

        # Validate period
        if period not in periods:
            period = 'during'

        start_date, end_date, period_title = periods[period]

        # === Expected keywords per period ===
        expected_keywords = {
            'before': ['deal', 'energy', 'market', 'project', 'stock', 'price', 'risk', 'contract'],
            'during': ['bankruptcy', 'fraud', 'loss', 'investigation', 'accounting', 'sec', 'audit', 'crisis'],
            'after': ['restructuring', 'lawsuit', 'court', 'compliance', 'settlement', 'penalty', 'oversight']
        }

        keywords = expected_keywords[period]

        # === Filter the dataset ===
        period_df = df[(df['parsed_date'] >= start_date) & (df['parsed_date'] <= end_date)]

        # If sample size is too big, randomly sample to avoid memory issues
        text_data = ' '.join(
            period_df['body_chunk'].astype(str).sample(n=5000, random_state=42)
            if len(period_df) > 5000 else period_df['body_chunk']
        )

        # === Custom stopwords ===
        stopwords = set(STOPWORDS)
        stopwords.update([
            'enron', 'com', 'subject', 'http', 'www', 'mail', 'sent', 'cc', 'forwarded',
            'pm', 'am', 're', 'fw', 'hou', 'ect', 'houston', 'attached', 'thanks', 'thank'
        ])

        # === Vectorize with only expected keywords ===
        vectorizer = CountVectorizer(vocabulary=keywords, stop_words='english')
        counts = vectorizer.fit_transform([text_data])
        word_freq = dict(zip(vectorizer.get_feature_names_out(), counts.toarray()[0]))

        # === Generate word cloud ===
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color='white',
            stopwords=stopwords,
            colormap='viridis',
            max_words=100,
            contour_width=2,
            contour_color='steelblue'
        ).generate_from_frequencies(word_freq)

        # === Plot ===
        plt.figure(figsize=(14, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Enron Email Keyword Cloud – {period_title}', fontsize=18, pad=20)
        plt.tight_layout()

        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return img_base64, f"Keyword Cloud: {period_title}", f"Expected topic keywords during {period_title.lower()}"

    except Exception as e:
        plt.close()
        raise Exception(f"Keyword cloud generation error: {str(e)}")


import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
import google.generativeai as genai
from datetime import datetime

# Configure API key (use your actual key)

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")

def story(request):
    narratives = [
        {'type': 'rise-business-model', 'title': 'Enron’s Rise and Business Model', 'date': '1985 - 2000'},
        {'type': 'executives-roles', 'title': 'Key Executives and Roles', 'date': '1990 - 2001'},
        {'type': 'trader', 'title': 'Energy Market Manipulation', 'date': '2000-2001'},
        {'type': 'whistleblower', 'title': 'Internal Warnings', 'date': 'August 2001'},
        {'type': 'accounting', 'title': 'Financial Collapse', 'date': 'October 2001'},
        {'type': 'legal', 'title': 'Bankruptcy Filing', 'date': 'December 2001'},
        {'type': 'special-purpose-entities', 'title': 'Special Purpose Entities (SPEs)', 'date': '1997 - 2001'},
        {'type': 'impact-employees-investors', 'title': 'Impact on Employees and Investors', 'date': '2001'},
        {'type': 'legal-trials', 'title': 'Legal Trials & Sentences', 'date': '2002 - 2006'},
        {'type': 'regulatory-changes', 'title': 'Regulatory Changes Post-Enron', 'date': '2002 - Present'},
    ]
    return render(request, 'enronviz/story.html', {'narratives': narratives})


def story_detail(request, story_type):
    narrative_lookup = {
        'rise-business-model': {
            'title': 'Enron’s Rise and Business Model',
            'date': '1985 - 2000',
            'content': 'How Enron grew and its innovative but risky business model.',
            'entities': ['Kenneth Lay (Founder/CEO)', 'Jeffrey Skilling (CEO)', 'Andrew Fastow (CFO)', 'Enron Corporation']
        },
        'executives-roles': {
            'title': 'Key Executives and Roles',
            'date': '1990 - 2001',
            'content': 'Profiles of major executives involved.',
            'entities': ['Kenneth Lay', 'Jeffrey Skilling', 'Andrew Fastow', 'Rebecca Mark', 'Lou Pai']
        },
        'trader': {
            'title': 'Energy Market Manipulation',
            'date': '2000-2001',
            'content': 'Details about energy market manipulation schemes.',
            'entities': ['Enron Energy Services', 'California electricity market', 'Traders: John Arnold', 'Traders: Tim Belden']
        },
        'whistleblower': {
            'title': 'Internal Warnings',
            'date': 'August 2001',
            'content': 'Accounts of whistleblowers and internal warnings.',
            'entities': ['Sherron Watkins', 'Vince Kaminski (Risk Analyst)', 'Enron Board of Directors']
        },
        'accounting': {
            'title': 'Financial Collapse',
            'date': 'October 2001',
            'content': 'The accounting practices leading to the collapse.',
            'entities': ['Arthur Andersen (Auditors)', 'Mark-to-Market accounting', 'Special Purpose Entities']
        },
        'legal': {
            'title': 'Bankruptcy Filing',
            'date': 'December 2001',
            'content': 'The bankruptcy filing and its implications.',
            'entities': ['U.S. Bankruptcy Court', 'Securities and Exchange Commission', 'Shareholders']
        },
        'special-purpose-entities': {
            'title': 'Special Purpose Entities (SPEs)',
            'date': '1997 - 2001',
            'content': 'The role of SPEs in hiding debt and liabilities.',
            'entities': ['LJM1 & LJM2 partnerships', 'Chewco Investments', 'Raptor vehicles', 'Andrew Fastow']
        },
        'impact-employees-investors': {
            'title': 'Impact on Employees and Investors',
            'date': '2001',
            'content': 'How the scandal affected workers and investors.',
            'entities': ['Enron employees', 'Pension funds', 'Shareholders', '401(k) plans']
        },
        'legal-trials': {
            'title': 'Legal Trials & Sentences',
            'date': '2002 - 2006',
            'content': 'Trials, convictions, and legal aftermath.',
            'entities': ['Department of Justice', 'Jeffrey Skilling', 'Kenneth Lay', 'Andrew Fastow', 'Arthur Andersen']
        },
        'regulatory-changes': {
            'title': 'Regulatory Changes Post-Enron',
            'date': '2002 - Present',
            'content': 'New laws and regulations enacted due to Enron.',
            'entities': ['Sarbanes-Oxley Act', 'Public Company Accounting Oversight Board', 'SEC reforms']
        },
    }

    narrative = narrative_lookup.get(story_type)
    if not narrative:
        return render(request, '404.html')  # Or raise Http404

    filtered = filter_emails(story_type)
    if len(filtered) == 0:
        return render(request, 'enronviz/story_detail.html', {
            'stories': [{
                'title': narrative['title'],
                'date': narrative['date'],
                'summary': "No emails found.",
                'emails': [],
                'entities': narrative['entities']  # Add entities here
            }]
        })

    sample_emails = filtered.sample(min(3, len(filtered)))
    prompt = f"""
    Create a concise 1-paragraph summary of the {narrative['title']} event during the Enron scandal.
    Focus on the key developments around {narrative['date']}.
    Use these email excerpts as context:
    {sample_emails['body_chunk'].tolist()}
    """

    response = model.generate_content(prompt)
    stories = [{
        'title': narrative['title'],
        'date': narrative['date'],
        'summary': response.text,
        'emails': format_emails(sample_emails),
        'entities': narrative['entities']  
    }]

    return render(request, 'enronviz/story_detail.html', {'stories': stories})

def filter_emails(narrative_type):
    """Filter emails based on narrative type with keyword matching."""
    if narrative_type == 'whistleblower':
        return df[df['rule_category'].str.lower() == 'legal']
    elif narrative_type == 'trader':
        return df[df['from'].str.contains('trader|trading', case=False, na=False)]
    elif narrative_type == 'accounting':
        return df[df['rule_category'].str.contains('accounting|finance', case=False, na=False)]
    elif narrative_type == 'legal':
        return df[df['rule_category'].str.lower() == 'legal']
    elif narrative_type == 'rise-business-model':
        return df[df['subject'].str.contains('business model|Enron rise|growth|energy trading', case=False, na=False)]
    elif narrative_type == 'executives-roles':
        return df[df['body_chunk'].str.contains('executive|CEO|Klaus|Fastow|Skilling|Lay', case=False, na=False)]
    elif narrative_type == 'special-purpose-entities':
        return df[df['body_chunk'].str.contains('SPE|special purpose entity|off balance sheet|debt hiding', case=False, na=False)]
    elif narrative_type == 'impact-employees-investors':
        return df[df['body_chunk'].str.contains('employee|investor|layoff|stock|retirement|benefits', case=False, na=False)]
    elif narrative_type == 'legal-trials':
        return df[df['body_chunk'].str.contains('trial|court|conviction|sentence|jury', case=False, na=False)]
    elif narrative_type == 'regulatory-changes':
        return df[df['body_chunk'].str.contains('Sarbanes|regulation|compliance|law|SEC|reform', case=False, na=False)]
    else:
        return pd.DataFrame()


def format_emails(email_df):
    """Format email data for template"""
    return [{
        'subject': email['body_chunk'][:50] + '...' if len(email['body_chunk']) > 50 else email['body_chunk'],
        'sender': email['from'],
        'recipient': email['to'],
        'date': email['date'],
        'content': email['body_chunk']
    } for _, email in email_df.iterrows()]


def generate_story_api(request):
    """API endpoint for external applications"""
    try:
        narrative_type = request.GET.get('narrative')
        filtered = filter_emails(narrative_type)

        if filtered.empty:
            return JsonResponse({'error': 'No data found'}, status=404)

        sample_emails = filtered.sample(min(3, len(filtered)))['body_chunk'].tolist()
        prompt = f"Generate narrative about {narrative_type} perspective using these emails: {sample_emails}"

        response = model.generate_content(prompt)
        return JsonResponse({
            'story': response.text,
            'source_emails': sample_emails
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest
import json
import re
@csrf_exempt
def generate_podcast(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed")

    try:
        data = json.loads(request.body.decode("utf-8"))
        title = data.get("title")
        summary = data.get("summary")

        if not title or not summary:
            return HttpResponseBadRequest("Missing title or summary")

        # Prompt for Gemini to generate a podcast
        prompt = f"""
You are generating a podcast transcript.

Title: "{title}"

Summary:
{summary}

Create a fictional podcast episode between a host and a guest.
- Use engaging and professional tone.
- Format clearly with speaker labels: "Host:", "Guest:".
- Include a warm intro and insightful conversation.
- Keep it under 300 words.
- Vary the voices and styles slightly.
"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        raw_transcript = response.text.strip()

        # Clean transcript by removing lines that look like stage directions or music notes
        cleaned_lines = []
        for line in raw_transcript.split('\n'):
            line = line.strip()
            # Remove lines with **...** or lines containing "music" or "fade" (case insensitive)
            if line and not re.search(r'^\*\*.*\*\*$|music|fade', line, re.I):
                cleaned_lines.append(line)

        cleaned_transcript = '\n'.join(cleaned_lines)

        return JsonResponse({
            "host": "Alex Johnson",
            "guest": "Former Enron Insider",
            "transcript": cleaned_transcript
        })

    except Exception as e:
        return HttpResponseBadRequest(f"Error: {str(e)}")

from django.core.cache import cache
import time
def all_stories(request):
    """Display all stories in a chronological timeline view with cached summaries and entities"""
    narratives = [
        {
            'type': 'rise-business-model', 
            'title': 'Enron’s Rise and Business Model', 
            'date': '1985 - 2000', 
            'sort_date': '1985',
            'entities': ['Kenneth Lay (Founder/CEO)', 'Jeffrey Skilling (CEO)', 'Andrew Fastow (CFO)', 'Enron Corporation']
        },
        {
            'type': 'executives-roles', 
            'title': 'Key Executives and Roles', 
            'date': '1990 - 2001', 
            'sort_date': '1990',
            'entities': ['Kenneth Lay', 'Jeffrey Skilling', 'Andrew Fastow', 'Rebecca Mark', 'Lou Pai']
        },
        {
            'type': 'special-purpose-entities', 
            'title': 'Special Purpose Entities (SPEs)', 
            'date': '1997 - 2001', 
            'sort_date': '1997',
            'entities': ['LJM1 & LJM2 partnerships', 'Chewco Investments', 'Raptor vehicles', 'Andrew Fastow']
        },
        {
            'type': 'trader', 
            'title': 'Energy Market Manipulation', 
            'date': '2000-2001', 
            'sort_date': '2000',
            'entities': ['Enron Energy Services', 'California electricity market', 'Traders: John Arnold', 'Traders: Tim Belden']
        },
        {
            'type': 'whistleblower', 
            'title': 'Internal Warnings', 
            'date': 'August 2001', 
            'sort_date': '2001-08',
            'entities': ['Sherron Watkins', 'Vince Kaminski (Risk Analyst)', 'Enron Board of Directors']
        },
        {
            'type': 'accounting', 
            'title': 'Financial Collapse', 
            'date': 'October 2001', 
            'sort_date': '2001-10',
            'entities': ['Arthur Andersen (Auditors)', 'Mark-to-Market accounting', 'Special Purpose Entities']
        },
        {
            'type': 'legal', 
            'title': 'Bankruptcy Filing', 
            'date': 'December 2001', 
            'sort_date': '2001-12',
            'entities': ['U.S. Bankruptcy Court', 'Securities and Exchange Commission', 'Shareholders']
        },
        {
            'type': 'impact-employees-investors', 
            'title': 'Impact on Employees and Investors', 
            'date': '2001', 
            'sort_date': '2001-12',
            'entities': ['Enron employees', 'Pension funds', 'Shareholders', '401(k) plans']
        },
        {
            'type': 'legal-trials', 
            'title': 'Legal Trials & Sentences', 
            'date': '2002 - 2006', 
            'sort_date': '2002',
            'entities': ['Department of Justice', 'Jeffrey Skilling', 'Kenneth Lay', 'Andrew Fastow', 'Arthur Andersen']
        },
        {
            'type': 'regulatory-changes', 
            'title': 'Regulatory Changes Post-Enron', 
            'date': '2002 - Present', 
            'sort_date': '2002',
            'entities': ['Sarbanes-Oxley Act', 'Public Company Accounting Oversight Board', 'SEC reforms']
        },
    ]
    
    # Sort by sort_date in ascending order
    narratives_sorted = sorted(narratives, key=lambda x: x['sort_date'])
    
    for narrative in narratives_sorted:
        try:
            # Check cache first for summary
            cache_key = f"story_summary_{narrative['type']}"
            cached_summary = cache.get(cache_key)
            
            if cached_summary:
                narrative['summary'] = cached_summary
                logger.info(f"Using cached summary for {narrative['title']}")
            else:
                # Only generate summary if not in cache
                filtered = filter_emails(narrative['type'])
                
                if len(filtered) > 0:
                    # Generate summary with error handling
                    try:
                        sample_emails = filtered.sample(min(3, len(filtered)))
                        email_texts = sample_emails['body_chunk'].tolist() if 'body_chunk' in sample_emails else []
                        
                        prompt = f"""
                        Create a concise 1-paragraph summary (3-5 sentences) of the {narrative['title']} event.
                        Focus on key developments around {narrative['date']}.
                        Context: {' '.join(str(text)[:500] for text in email_texts)}
                        """
                        
                        # Rate limit API calls
                        time.sleep(1)  # 1 second delay between calls
                        response = model.generate_content(prompt)
                        narrative['summary'] = response.text
                        
                        # Cache the summary for 24 hours (86400 seconds)
                        cache.set(cache_key, narrative['summary'], 86400)
                        logger.info(f"Generated and cached new summary for {narrative['title']}")
                    except Exception as e:
                        logger.error(f"Error generating summary for {narrative['title']}: {str(e)}")
                        narrative['summary'] = get_fallback_summary(narrative)
                else:
                    narrative['summary'] = get_fallback_summary(narrative)
                
                # Always process emails (not cached)
                narrative['emails'] = get_email_samples(filtered)
                
        except Exception as e:
            logger.error(f"Error processing narrative {narrative['title']}: {str(e)}")
            narrative['summary'] = get_fallback_summary(narrative)
            narrative['emails'] = []
    
    return render(request, 'enronviz/all_stories.html', {
        'narratives': narratives_sorted,
        'timeline_view': True
    })

def get_email_samples(filtered_emails, sample_size=3):
    """Extract sample emails from filtered results"""
    emails = []
    if len(filtered_emails) > 0:
        for idx, row in filtered_emails.head(sample_size).iterrows():
            emails.append({
                'subject': row.get('subject', 'No subject'),
                'sender': row.get('from', 'Unknown'),
                'recipient': row.get('to', 'Unknown'),
                'date': row.get('parsed_date', 'Unknown').strftime('%B %d, %Y') if pd.notnull(row.get('parsed_date')) else 'Unknown',
                'content': (row.get('body_chunk', '')[:200] + '...') if isinstance(row.get('body_chunk', ''), str) else ''
            })
    return emails

def get_fallback_summary(narrative):
    """Generate a fallback summary when API fails or no emails exist"""
    fallbacks = {
        'rise-business-model': f"Enron's innovative but risky business model propelled its rapid growth from {narrative['date']}.",
        'executives-roles': f"Key executives shaped Enron's corporate culture and strategic direction during {narrative['date']}.",
        'special-purpose-entities': f"SPEs were used to hide debt and inflate profits {narrative['date']}, contributing to the scandal.",
        'trader': f"Energy market manipulation through trading strategies occurred {narrative['date']}.",
        'whistleblower': f"Internal warnings about accounting practices emerged in {narrative['date']}.",
        'accounting': f"Financial irregularities led to Enron's collapse in {narrative['date']}.",
        'legal': f"Enron filed for bankruptcy protection in {narrative['date']}.",
        'impact-employees-investors': f"Many employees and investors suffered significant losses in {narrative['date']}.",
        'legal-trials': f"Key figures faced legal consequences between {narrative['date']}.",
        'regulatory-changes': f"Major regulatory reforms were implemented after {narrative['date']}."
    }
    return fallbacks.get(narrative['type'], f"This event occurred during {narrative['date']}.")
# Add these imports at the top if not already present
import json
import re
import pytz
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from datetime import datetime, timedelta
import pandas as pd
import google.generativeai as genai
from django.conf import settings
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Timezone setup
PACIFIC_TZ = pytz.timezone('US/Pacific')
UTC_TZ = pytz.UTC

@csrf_exempt
def chat_with_agent(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            if not user_message:
                return JsonResponse({'error': 'Empty message'}, status=400)
            
            logger.info(f"Received query: {user_message}")
            
            # Step 1: Search relevant emails from your dataset
            relevant_emails = search_emails(user_message)
            
            # Step 2: Generate response using Gemini
            response = generate_chat_response(user_message, relevant_emails)
            
            return JsonResponse({'response': response})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return JsonResponse({'error': 'Internal server error'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def search_emails(query):
    """Enhanced email search with proper timezone handling"""
    try:
        # First try to extract date references
        date_ref = extract_date_reference(query)
        
        base_query = df.copy()
        
        # Ensure parsed_date is properly converted to datetime with UTC timezone
        if not pd.api.types.is_datetime64_any_dtype(base_query['parsed_date']):
            base_query['parsed_date'] = pd.to_datetime(base_query['parsed_date'], utc=True)
        elif base_query['parsed_date'].dt.tz is None:
            base_query['parsed_date'] = base_query['parsed_date'].dt.tz_localize('UTC')
        
        # Log dataset date range for debugging
        logger.info(f"Dataset date range: {base_query['parsed_date'].min()} to {base_query['parsed_date'].max()}")
        
        if date_ref:
            # Ensure date_ref values are timezone-aware
            if date_ref['start'].tzinfo is None:
                date_ref['start'] = date_ref['start'].replace(tzinfo=UTC_TZ)
            if date_ref['end'].tzinfo is None:
                date_ref['end'] = date_ref['end'].replace(tzinfo=UTC_TZ)
            
            logger.info(f"Date filter range: {date_ref['start']} to {date_ref['end']}")
            
            # Filter by date range - using inclusive bounds
            base_query = base_query[
                (base_query['parsed_date'] >= date_ref['start']) & 
                (base_query['parsed_date'] <= date_ref['end'])
            ]
            
            logger.info(f"Found {len(base_query)} emails in date range")
        
        # Rest of search logic remains the same...
        people_terms = ['ken', 'kenneth', 'lay', 'skilling', 'jeff', 'andrew', 'fastow']
        is_person_query = any(term in query.lower() for term in people_terms)
        
        if is_person_query:
            person_name = extract_name(query)
            results = base_query[
                (base_query['from'].str.contains(person_name, case=False, na=False)) |
                (base_query['to'].str.contains(person_name, case=False, na=False))
            ]
        else:
            # Split query into terms for better matching
            query_terms = re.findall(r'\w+', query.lower())
            conditions = []
            
            for term in query_terms:
                if len(term) > 3:  # Only search for terms longer than 3 characters
                    conditions.append(
                        (base_query['subject'].str.contains(term, case=False, na=False)) |
                        (base_query['body_chunk'].str.contains(term, case=False, na=False)) |
                        (base_query['topic_label'].str.contains(term, case=False, na=False)) |
                        (base_query['rule_category'].str.contains(term, case=False, na=False))
                    )
            
            if conditions:
                combined_condition = conditions[0]
                for cond in conditions[1:]:
                    combined_condition |= cond
                results = base_query[combined_condition]
            else:
                results = base_query
        
        # If no results but we had a date filter, show nearest dates
        if len(results) == 0 and date_ref:
            # Get all emails from the dataset (ignoring previous filters)
            all_emails = df.copy()
            all_emails['parsed_date'] = pd.to_datetime(all_emails['parsed_date'], utc=True)
            
            # Find nearest dates
            nearest_before = all_emails[all_emails['parsed_date'] < date_ref['start']].sort_values('parsed_date', ascending=False).head(1)
            nearest_after = all_emails[all_emails['parsed_date'] > date_ref['end']].sort_values('parsed_date').head(1)
            
            date_suggestion = ""
            if len(nearest_before) > 0:
                date_suggestion += f" Closest earlier email: {nearest_before['parsed_date'].iloc[0].strftime('%B %d, %Y')}"
            if len(nearest_after) > 0:
                date_suggestion += f" Closest later email: {nearest_after['parsed_date'].iloc[0].strftime('%B %d, %Y')}"
            
            if date_suggestion:
                return pd.DataFrame([{
                    'subject': 'Date range suggestion',
                    'body_chunk': f"No emails found in exact date range.{date_suggestion}",
                    'parsed_date': datetime.now(UTC_TZ),
                    'from': 'System',
                    'to': 'User',
                    'topic_label': 'Date Information',
                    'urgency_label': 'Low'
                }])
        
        return results.sort_values('parsed_date', ascending=False).head(5)
    
    except Exception as e:
        logger.error(f"Error in search_emails: {str(e)}", exc_info=True)
        return pd.DataFrame()  # Return empty DataFrame on error

def extract_date_reference(query):
    """Enhanced date extraction with consistent timezone handling"""
    try:
        now = datetime.now(UTC_TZ)
        query = query.lower()
        
        # First check for specific year mentions (2000, 2001, etc.)
        year_match = re.search(r'(19\d{2}|20\d{2})', query)
        if year_match:
            year = int(year_match.group(1))
            start = PACIFIC_TZ.localize(datetime(year, 1, 1)).astimezone(UTC_TZ)
            end = PACIFIC_TZ.localize(datetime(year, 12, 31, 23, 59, 59)).astimezone(UTC_TZ)
            return {'start': start, 'end': end}
        
        # Relative dates
        if 'today' in query:
            start = datetime(now.year, now.month, now.day, tzinfo=UTC_TZ)
            end = start + timedelta(days=1)
            return {'start': start, 'end': end}
        elif 'yesterday' in query:
            start = datetime(now.year, now.month, now.day, tzinfo=UTC_TZ) - timedelta(days=1)
            end = start + timedelta(days=1)
            return {'start': start, 'end': end}
        elif 'last week' in query:
            start = datetime(now.year, now.month, now.day, tzinfo=UTC_TZ) - timedelta(days=now.weekday() + 7)
            end = start + timedelta(days=7)
            return {'start': start, 'end': end}
        
        # Specific date patterns
        date_patterns = [
            (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),  # YYYY-MM-DD
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%m/%d/%Y'),  # MM/DD/YYYY
            (r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{1,2}),? (\d{4})', '%b %d %Y'),  # Month Day, Year
            (r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{4})', '%b %Y'),  # Month Year
            (r'(\d{1,2}) (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{4})', '%d %b %Y'),  # Day Month Year
        ]
        
        for pattern, fmt in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    date_str = ' '.join([g for g in match.groups() if g])
                    naive_date = datetime.strptime(date_str, fmt)
                    # Convert to timezone-aware in Pacific time, then to UTC
                    local_date = PACIFIC_TZ.localize(naive_date)
                    utc_date = local_date.astimezone(UTC_TZ)
                    
                    if fmt in ('%Y-%m-%d', '%m/%d/%Y', '%b %d %Y', '%d %b %Y'):
                        end_date = utc_date + timedelta(days=1)
                    elif fmt == '%b %Y':
                        if naive_date.month == 12:
                            end_date = PACIFIC_TZ.localize(datetime(naive_date.year+1, 1, 1)).astimezone(UTC_TZ)
                        else:
                            end_date = PACIFIC_TZ.localize(datetime(naive_date.year, naive_date.month+1, 1)).astimezone(UTC_TZ)
                    return {'start': utc_date, 'end': end_date}
                except ValueError as e:
                    logger.warning(f"Date parsing error for pattern {pattern}: {str(e)}")
                    continue
        
        # Month-year patterns
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month, num in month_map.items():
            if month in query:
                year_match = re.search(r'(\d{4})', query)
                year = int(year_match.group(1)) if year_match else now.year
                start = PACIFIC_TZ.localize(datetime(year, num, 1)).astimezone(UTC_TZ)
                if num == 12:
                    end = PACIFIC_TZ.localize(datetime(year+1, 1, 1)).astimezone(UTC_TZ)
                else:
                    end = PACIFIC_TZ.localize(datetime(year, num+1, 1)).astimezone(UTC_TZ)
                return {'start': start, 'end': end}
        
        return None
    
    except Exception as e:
        logger.error(f"Error in extract_date_reference: {str(e)}", exc_info=True)
        return None

def generate_chat_response(user_query, relevant_emails):
    """Generate response with proper error handling"""
    try:
        date_ref = extract_date_reference(user_query)
        date_context = ""
        
        if date_ref:
            # Format dates for display in Pacific time (original email timezone)
            start_pacific = date_ref['start'].astimezone(PACIFIC_TZ)
            end_pacific = date_ref['end'].astimezone(PACIFIC_TZ)
            date_context = f"\nTime Period: {start_pacific.strftime('%B %d, %Y')} to {end_pacific.strftime('%B %d, %Y')}"
        
        # Handle case where no emails were found but we have date suggestions
        if len(relevant_emails) == 1 and relevant_emails.iloc[0]['subject'] == 'Date range suggestion':
            return relevant_emails.iloc[0]['body_chunk']
        
        if len(relevant_emails) == 0:
            if date_ref:
                return f"No emails found in the specified date range: {start_pacific.strftime('%B %d, %Y')} to {end_pacific.strftime('%B %d, %Y')}"
            else:
                return "No emails found matching your query."
        
        context = f"""
        You are analyzing the Enron email dataset. {date_context}
        User query: {user_query}
        
        Relevant emails found:
        {json.dumps([
            {
                'date': email['parsed_date'].astimezone(PACIFIC_TZ).strftime('%B %d, %Y %I:%M %p'),
                'from': email['from'],
                'to': email['to'],
                'subject': email['subject'],
                'topic': email['topic_label'],
                'urgency': email['urgency_label'],
                'excerpt': email['body_chunk'][:200] + '...' if len(email['body_chunk']) > 200 else email['body_chunk']
            }
            for _, email in relevant_emails.iterrows()
        ], indent=2)}
        """
        
        prompt = f"""
        When responding:
        1. Always mention specific dates in format "Month Day, Year" (e.g., "November 8, 2000")
        2. For time-specific queries, organize chronologically
        3. Highlight patterns in topics, urgency, or rule categories
        4. Maintain professional but conversational tone
        5. If the query is specifically about a date range, begin by confirming the date range you're reporting on
        
        Context: {context}
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        logger.error(f"Error in generate_chat_response: {str(e)}", exc_info=True)
        return "Sorry, I encountered an error while processing your request."



def extract_name(query):
    """Extract likely person name from query with error handling"""
    try:
        people_mapping = {
            'ken': 'kenneth.lay', 'kenneth': 'kenneth.lay', 'lay': 'kenneth.lay',
            'jeff': 'jeff.skilling', 'skilling': 'jeff.skilling',
            'andrew': 'andrew.fastow', 'fastow': 'andrew.fastow',
            'rebecca': 'rebecca.mark', 'mark': 'rebecca.mark',
            'lou': 'lou.pai', 'pai': 'lou.pai',
            'greg': 'greg.whalley', 'whalley': 'greg.whalley'
        }
        
        query = query.lower()
        for term, email in people_mapping.items():
            if term in query:
                return email
        return query.split()[0].lower()
    except Exception as e:
        logger.error(f"Error in extract_name: {str(e)}", exc_info=True)
        return ""