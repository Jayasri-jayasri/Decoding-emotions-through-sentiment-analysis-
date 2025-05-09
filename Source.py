import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv("C:/Users/USER/Downloads/sentiment140dataset.csv", encoding='ISO-8859-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Basic text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#','', text)  # Remove @ and #
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuations
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply cleaning function to the text column
df['clean_text'] = df['text'].apply(clean_text)

# Encode the target labels
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['target'].astype(str))

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
