import nltk
import re
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# تنظيف النصوص
def clean_text(text):
    if not text or not text.strip():
        return ""
    text = text.lower()
    # إزالة علامات الترقيم والرموز غير الحروف والأرقام باستبدالها بمسافة
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

#تحويل النصوص النظيفه لمصفوفة ارقام
def vectorize_text(cleaned_texts):
     vectorizer = CountVectorizer()
     X = vectorizer.fit_transform(cleaned_texts)
     return X.toarray(), vectorizer

# كود بناء الشبكة العصبونيه
def build_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dense(1, activation='sigmoid')
         ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#التشغيل
if __name__ == "__main__":

    # CSVقراءة ملف ال  
    df = pd.read_csv("feedback.csv")
     # تنظيف النصوص باستخدام الدالة clean_text
    cleaned_texts = df['text'].apply(clean_text)
      # تحويل النصوص إلى Bag of Words
    X, vectorizer = vectorize_text(cleaned_texts)
     # تحضير التصنيفات كـ numpy array
    labels = df['label'].values
    # بناء النموذج
    model = build_model(X.shape[1])
    model.summary()
    # تدريب النموذج
    model.fit(X, labels, epochs=20, batch_size=4, verbose=1)
    # تقييم النموذج
    loss, accuracy = model.evaluate(X, labels, verbose=0)
    print("Model Accuracy:", accuracy)
 
# حفظ النموذج والمتجه
    #model.save("sentiment_model.h5")
   # joblib.dump(vectorizer, "vectorizer.pkl")

 # تجريب النموذج على جمل جديدة
test_sentences = [
    "I really love the support I received",  # إيجابي
    "This was a horrible experience",        # سلبي
    "Very friendly and helpful team",        # إيجابي
    "I'm never buying from here again"       # سلبي
]
# تنظيف الجمل الجديدة
cleaned_test = [clean_text(text) for text in test_sentences]
# تحويلها إلى Bag of Words بنفس الـ vectorizer المستخدم في التدريب
X_test_new = vectorizer.transform(cleaned_test)
# التنبؤ
predictions = model.predict(X_test_new)
# عرض النتائج
for sentence, prediction in zip(test_sentences, predictions):
    label = "Positive" if prediction[0] > 0.5 else "Negative"
    print(f"\"{sentence}\" → {label}")


        

      