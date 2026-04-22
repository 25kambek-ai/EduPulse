import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================= DATA =================
feedback_data = [
    {"rating": 5, "comment": "The teacher explains concepts very clearly and is helpful."},
    {"rating": 4, "comment": "Good teaching but sometimes too fast."},
    {"rating": 2, "comment": "Hard to understand and not enough examples."},
    {"rating": 3, "comment": "Average teaching, communication could improve."},
    {"rating": 5, "comment": "Very engaging and interactive class."},
    {"rating": 1, "comment": "Teacher is rude and not supportive."},
    {"rating": 4, "comment": "Good explanation but a bit fast."},
    {"rating": 2, "comment": "Lectures are boring and difficult."},
    {"rating": 5, "comment": "Amazing teaching style, very clear."},
    {"rating": 3, "comment": "Okay experience, needs improvement."}
]

# ================= SENTIMENT =================
def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df = pd.DataFrame(feedback_data)
df['sentiment'] = df['rating'].apply(label_sentiment)

# ================= NLP MODEL =================
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['comment'])
y = df['sentiment']

model = LogisticRegression()
model.fit(X, y)

df['predicted'] = model.predict(X)

# ================= ANALYSIS =================
total = len(df)
positive = (df['predicted'] == "Positive").sum()
neutral = (df['predicted'] == "Neutral").sum()
negative = (df['predicted'] == "Negative").sum()

# ================= THEMES =================
themes = []

if df['comment'].str.contains("clear|explain", case=False).any():
    themes.append({"theme": "Teaching Clarity", "description": "Clarity of explanation", "frequency": "High"})

if df['comment'].str.contains("fast", case=False).any():
    themes.append({"theme": "Pacing", "description": "Lecture speed issues", "frequency": "Medium"})

if df['comment'].str.contains("rude", case=False).any():
    themes.append({"theme": "Behavior", "description": "Teacher attitude", "frequency": "Low"})

if df['comment'].str.contains("boring|engaging", case=False).any():
    themes.append({"theme": "Engagement", "description": "Student engagement level", "frequency": "Medium"})

# ================= STRENGTHS & ISSUES =================
strengths = ["Clear explanations", "Engaging teaching", "Helpful attitude"]
improvements = ["Lecture pacing too fast", "Behavior improvement needed", "More examples required"]

# ================= QUOTES =================
quotes = df['comment'].sample(3).tolist()

# ================= FINAL OUTPUT =================
output = {
    "overall_sentiment": {
        "positive_percentage": round((positive/total)*100, 2),
        "neutral_percentage": round((neutral/total)*100, 2),
        "negative_percentage": round((negative/total)*100, 2),
        "summary": "Overall feedback is mixed with strong positives in clarity and engagement, but concerns in pacing and behavior."
    },
    "key_themes": themes,
    "strengths": strengths,
    "areas_for_improvement": improvements,
    "representative_quotes": quotes,
    "recommendations": [
        "Slow down lecture pace",
        "Improve interaction with students",
        "Maintain positive behavior",
        "Provide more examples for clarity"
    ]
}

print("\nFINAL REPORT:\n")
print(json.dumps(output, indent=4))

# ================= CHATBOT =================
def chatbot(query):
    query = query.lower()

    if "sentiment" in query:
        return output["overall_sentiment"]
    elif "issues" in query:
        return output["areas_for_improvement"]
    elif "like" in query:
        return output["strengths"]
    elif "recommend" in query:
        return output["recommendations"]
    elif "summary" in query:
        return output["overall_sentiment"]["summary"]
    else:
        return "Ask about sentiment, issues, strengths, or recommendations."

print("\nCHATBOT TEST:")
print(chatbot("what are issues"))
