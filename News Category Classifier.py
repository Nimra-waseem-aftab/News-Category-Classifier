import pandas as pd
import numpy as np
import re
import pickle
import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pyperclip
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Global variables
model = None
vectorizer = None
preprocessor = None
model_accuracy = 0.0
model_name = ""

# Text Preprocessor
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.extra_stopwords = {'said', 'would', 'could', 'also', 'one', 'two', 'like', 'get', 'make', 'new', 'news'}
        self.stop_words.update(self.extra_stopwords)
    
    def clean_text(self, text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)

# Clean and validate dataset
def clean_dataset(df):
    print("Cleaning dataset...")
    
    column_mapping = {}
    actual_columns = df.columns.tolist()
    print(f"Actual columns in dataset: {actual_columns}")
    
    mapping_rules = {
        'headline': ['headline', 'title', 'news'],
        'category': ['category', 'categories', 'type'],
        'short_description': ['short_description', 'description', 'summary'],
        'authors': ['authors', 'author', 'writer', 'byline'],
        'link': ['link', 'url', 'news_link'],
        'date': ['date', 'published_date', 'timestamp']
    }
    
    for standard_name, possible_names in mapping_rules.items():
        for col in actual_columns:
            if col.lower() in possible_names or any(name in col.lower() for name in possible_names):
                column_mapping[standard_name] = col
                break
    
    if column_mapping:
        df = df.rename(columns={v: k for k, v in column_mapping.items()})
        print(f"Renamed columns: {column_mapping}")
    
    essential_cols = ['headline', 'category', 'short_description']
    missing_essential = [col for col in essential_cols if col not in df.columns]
    
    if missing_essential:
        print(f"Warning: Missing essential columns: {missing_essential}")
        if len(df.columns) >= 3:
            if 'headline' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'headline'})
            if 'category' not in df.columns:
                df = df.rename(columns={df.columns[1]: 'category'})
            if 'short_description' not in df.columns:
                df = df.rename(columns={df.columns[2]: 'short_description'})
    
    if 'headline' in df.columns:
        df['headline'] = df['headline'].fillna('')
    else:
        df['headline'] = ''
    
    if 'short_description' in df.columns:
        df['short_description'] = df['short_description'].fillna('')
    else:
        df['short_description'] = ''
    
    if 'authors' in df.columns:
        df['authors'] = df['authors'].fillna('')
    else:
        df['authors'] = ''
    
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('UNKNOWN')
    else:
        print("Error: No category column found!")
        return pd.DataFrame()
    
    df['category'] = df['category'].astype(str).str.strip().str.upper()
    df = df[df['category'] != '']
    df = df[df['category'] != 'UNKNOWN']
    
    mask_both_empty = (df['headline'].str.strip() == '') & (df['short_description'].str.strip() == '')
    df = df[~mask_both_empty]
    
    text_parts = []
    if 'headline' in df.columns:
        text_parts.append(df['headline'].str.strip())
    if 'short_description' in df.columns:
        text_parts.append(df['short_description'].str.strip())
    if 'authors' in df.columns:
        text_parts.append(df['authors'].str.strip())
    
    if text_parts:
        df['combined_text'] = text_parts[0]
        for part in text_parts[1:]:
            df['combined_text'] = df['combined_text'] + ' ' + part
    else:
        df['combined_text'] = ''
    
    df = df[df['combined_text'].str.strip() != '']
    
    category_counts = df['category'].value_counts()
    print(f"Total unique categories before filtering: {len(category_counts)}")
    
    min_samples = 10
    valid_categories = category_counts[category_counts >= min_samples].index
    df = df[df['category'].isin(valid_categories)]
    
    max_categories = 20
    if len(valid_categories) > max_categories:
        top_categories = category_counts.head(max_categories).index
        df = df[df['category'].isin(top_categories)]
    
    print(f"Categories after cleaning: {len(df['category'].unique())}")
    print(f"Final dataset size: {len(df)} articles")
    
    return df

# Load dataset
def load_dataset():
    print("Loading dataset...")
    
    possible_files = ['News_Category_Dataset_v3.json', 'news_category.json', 'News_Category.csv']
    file_found = None
    
    for file in possible_files:
        if os.path.exists(file):
            file_found = file
            break
    
    if not file_found:
        print("Error: Dataset file not found.")
        return None
    
    print(f"Found file: {file_found}")
    
    try:
        if file_found.endswith('.json'):
            data = []
            with open(file_found, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            df = pd.DataFrame(data)
        elif file_found.endswith('.csv'):
            df = pd.read_csv(file_found)
        else:
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    df = clean_dataset(df)
    
    if len(df) == 0:
        return None
    
    return df

# Train models
def train_models(df):
    global preprocessor
    print("Training models...")
    
    preprocessor = TextPreprocessor()
    df['processed_text'] = df['combined_text'].apply(preprocessor.preprocess)
    df = df[df['processed_text'].str.strip() != '']
    
    X = df['processed_text']
    y = df['category']
    
    category_counts = y.value_counts()
    min_samples_per_category = category_counts.min()
    
    if min_samples_per_category < 2:
        stratify_param = None
    else:
        stratify_param = y
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_vec, y_train)
    nb_pred = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    
    try:
        lr_model = LogisticRegression(max_iter=500, random_state=42, C=1.0, solver='lbfgs')
        lr_model.fit(X_train_vec, y_train)
        lr_pred = lr_model.predict(X_test_vec)
        lr_acc = accuracy_score(y_test, lr_pred)
        print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    except Exception:
        lr_acc = 0
        lr_model = None
    
    if lr_model is not None and lr_acc > nb_acc:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_acc = lr_acc
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        best_acc = nb_acc
    
    print(f"Best Model: {best_name} (Accuracy: {best_acc:.4f})")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('model_info.pkl', 'wb') as f:
        pickle.dump({'name': best_name, 'accuracy': best_acc}, f)
    
    return vectorizer, preprocessor, best_model, best_name, best_acc

# Desktop Popup Application
class NewsClassifierPopup:
    def __init__(self, root):
        self.root = root
        self.root.title("News Classifier - Always On Top")
        self.root.geometry("600x700")
        
        # Make window always on top
        self.root.attributes('-topmost', True)
        
        # Make window stay on top even when other apps are focused
        self.root.wm_attributes("-topmost", 1)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#667eea", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🔍 News Category Classifier",
            font=("Arial", 18, "bold"),
            bg="#667eea",
            fg="white"
        )
        title_label.pack(pady=10)
        
        info_label = tk.Label(
            header_frame,
            text=f"Model: {model_name} | Accuracy: {round(model_accuracy*100, 2)}%",
            font=("Arial", 10),
            bg="#667eea",
            fg="white"
        )
        info_label.pack()
        
        # Instructions Frame
        inst_frame = tk.Frame(self.root, bg="#f0f0f0")
        inst_frame.pack(fill=tk.X, padx=10, pady=10)
        
        inst_label = tk.Label(
            inst_frame,
            text="📋 Copy any news text from the internet, then click 'Paste & Classify' or press Ctrl+V",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#333",
            wraplength=560,
            justify=tk.LEFT
        )
        inst_label.pack(pady=5)
        
        # Input Frame
        input_frame = tk.Frame(self.root, bg="white")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(
            input_frame,
            text="News Text:",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        # Text input with scrollbar
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            height=10,
            bg="#fafafa",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button Frame
        button_frame = tk.Frame(self.root, bg="white")
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Paste & Classify Button
        paste_btn = tk.Button(
            button_frame,
            text="📋 Paste & Classify",
            font=("Arial", 12, "bold"),
            bg="#667eea",
            fg="white",
            activebackground="#5568d3",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.paste_and_classify
        )
        paste_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Classify Button
        classify_btn = tk.Button(
            button_frame,
            text="🚀 Classify",
            font=("Arial", 12, "bold"),
            bg="#764ba2",
            fg="white",
            activebackground="#6a3d91",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.classify_text
        )
        classify_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Clear Button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=("Arial", 12, "bold"),
            bg="#95a5a6",
            fg="white",
            activebackground="#7f8c8d",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.clear_all
        )
        clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Result Frame
        result_frame = tk.Frame(self.root, bg="white")
        result_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        tk.Label(
            result_frame,
            text="Classification Result:",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        # Result display
        self.result_frame_inner = tk.Frame(result_frame, bg="white")
        self.result_frame_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_label = tk.Label(
            self.result_frame_inner,
            text="Paste or type news text above and click 'Classify'",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666",
            wraplength=560,
            justify=tk.CENTER,
            pady=20
        )
        self.result_label.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#ecf0f1", height=30)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(
            footer_frame,
            text="💡 This window stays on top of all apps | Press ESC to minimize",
            font=("Arial", 8),
            bg="#ecf0f1",
            fg="#666"
        )
        footer_label.pack(pady=5)
        
        # Keyboard shortcuts
        self.root.bind('<Control-v>', lambda e: self.paste_and_classify())
        self.root.bind('<Return>', lambda e: self.classify_text())
        self.root.bind('<Escape>', lambda e: self.root.iconify())
        
    def paste_and_classify(self):
        try:
            clipboard_text = pyperclip.paste()
            if clipboard_text:
                self.text_input.delete(1.0, tk.END)
                self.text_input.insert(1.0, clipboard_text)
                self.classify_text()
            else:
                messagebox.showwarning("Empty Clipboard", "No text found in clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not paste from clipboard: {str(e)}")
    
    def classify_text(self):
        text = self.text_input.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Empty Input", "Please enter some news text to classify!")
            return
        
        if len(text) < 10:
            messagebox.showwarning("Text Too Short", "Please enter more text for accurate classification!")
            return
        
        try:
            # Preprocess and classify
            processed = preprocessor.preprocess(text)
            
            if not processed or len(processed) < 5:
                messagebox.showwarning("Invalid Text", "The text doesn't contain enough meaningful words!")
                return
            
            vectorized = vectorizer.transform([processed])
            prediction = model.predict(vectorized)[0]
            confidence_scores = model.predict_proba(vectorized)[0]
            confidence = np.max(confidence_scores) * 100
            
            # Get top 3 predictions
            top_indices = np.argsort(confidence_scores)[-3:][::-1]
            top_categories = model.classes_[top_indices]
            top_confidences = confidence_scores[top_indices] * 100
            
            # Display result
            self.display_result(prediction, confidence, top_categories, top_confidences)
            
        except Exception as e:
            messagebox.showerror("Classification Error", f"Error: {str(e)}")
    
    def display_result(self, category, confidence, top_categories, top_confidences):
        # Clear previous result
        for widget in self.result_frame_inner.winfo_children():
            widget.destroy()
        
        # Main result
        main_frame = tk.Frame(self.result_frame_inner, bg="#667eea", relief=tk.RAISED, borderwidth=2)
        main_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            main_frame,
            text="📰 PREDICTED CATEGORY",
            font=("Arial", 10, "bold"),
            bg="#667eea",
            fg="white"
        ).pack(pady=(10, 5))
        
        tk.Label(
            main_frame,
            text=category,
            font=("Arial", 20, "bold"),
            bg="#667eea",
            fg="white"
        ).pack(pady=5)
        
        # Confidence indicator
        if confidence > 80:
            conf_color = "#4caf50"
            conf_text = "High Confidence"
        elif confidence > 60:
            conf_color = "#ff9800"
            conf_text = "Medium Confidence"
        else:
            conf_color = "#f44336"
            conf_text = "Low Confidence"
        
        conf_frame = tk.Frame(main_frame, bg=conf_color)
        conf_frame.pack(fill=tk.X, pady=(5, 10))
        
        tk.Label(
            conf_frame,
            text=f"{conf_text}: {confidence:.2f}%",
            font=("Arial", 12, "bold"),
            bg=conf_color,
            fg="white"
        ).pack(pady=5)
        
        # Top 3 predictions
        top3_frame = tk.Frame(self.result_frame_inner, bg="white")
        top3_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            top3_frame,
            text="Top 3 Predictions:",
            font=("Arial", 10, "bold"),
            bg="white"
        ).pack(anchor=tk.W, pady=(5, 5))
        
        for i, (cat, conf) in enumerate(zip(top_categories, top_confidences), 1):
            pred_frame = tk.Frame(top3_frame, bg="#f0f0f0", relief=tk.SOLID, borderwidth=1)
            pred_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(
                pred_frame,
                text=f"{i}. {cat}",
                font=("Arial", 10, "bold"),
                bg="#f0f0f0",
                anchor=tk.W
            ).pack(side=tk.LEFT, padx=10, pady=5)
            
            tk.Label(
                pred_frame,
                text=f"{conf:.2f}%",
                font=("Arial", 10),
                bg="#f0f0f0",
                fg="#667eea",
                anchor=tk.E
            ).pack(side=tk.RIGHT, padx=10, pady=5)
    
    def clear_all(self):
        self.text_input.delete(1.0, tk.END)
        
        for widget in self.result_frame_inner.winfo_children():
            widget.destroy()
        
        self.result_label = tk.Label(
            self.result_frame_inner,
            text="Paste or type news text above and click 'Classify'",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666",
            wraplength=560,
            justify=tk.CENTER,
            pady=20
        )
        self.result_label.pack(fill=tk.BOTH, expand=True)

# Main execution
def main():
    global model, vectorizer, preprocessor, model_name, model_accuracy
    
    print("="*60)
    print("NEWS CLASSIFICATION SYSTEM - DESKTOP POPUP")
    print("="*60)
    
    # Load dataset
    df = load_dataset()
    
    if df is None or len(df) == 0:
        print("Error: Could not load dataset.")
        return
    
    # Train or load models
    try:
        if not all(os.path.exists(f) for f in ['model.pkl', 'vectorizer.pkl', 'preprocessor.pkl']):
            vectorizer, preprocessor, model, model_name, model_accuracy = train_models(df)
        else:
            print("Loading trained models...")
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('model_info.pkl', 'rb') as f:
                info = pickle.load(f)
                model_name = info['name']
                model_accuracy = info['accuracy']
    except Exception as e:
        print(f"Error: {e}")
        vectorizer, preprocessor, model, model_name, model_accuracy = train_models(df)
    
    print("="*60)
    print("Launching Desktop Popup Application...")
    print("Window will stay on top of all applications")
    print("="*60)
    
    # Launch GUI
    root = tk.Tk()
    app = NewsClassifierPopup(root)
    root.mainloop()

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import re
import pickle
import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pyperclip
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Global variables
model = None
vectorizer = None
preprocessor = None
model_accuracy = 0.0
model_name = ""

# Text Preprocessor
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.extra_stopwords = {'said', 'would', 'could', 'also', 'one', 'two', 'like', 'get', 'make', 'new', 'news'}
        self.stop_words.update(self.extra_stopwords)
    
    def clean_text(self, text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)

# Clean and validate dataset
def clean_dataset(df):
    print("Cleaning dataset...")
    
    column_mapping = {}
    actual_columns = df.columns.tolist()
    print(f"Actual columns in dataset: {actual_columns}")
    
    mapping_rules = {
        'headline': ['headline', 'title', 'news'],
        'category': ['category', 'categories', 'type'],
        'short_description': ['short_description', 'description', 'summary'],
        'authors': ['authors', 'author', 'writer', 'byline'],
        'link': ['link', 'url', 'news_link'],
        'date': ['date', 'published_date', 'timestamp']
    }
    
    for standard_name, possible_names in mapping_rules.items():
        for col in actual_columns:
            if col.lower() in possible_names or any(name in col.lower() for name in possible_names):
                column_mapping[standard_name] = col
                break
    
    if column_mapping:
        df = df.rename(columns={v: k for k, v in column_mapping.items()})
        print(f"Renamed columns: {column_mapping}")
    
    essential_cols = ['headline', 'category', 'short_description']
    missing_essential = [col for col in essential_cols if col not in df.columns]
    
    if missing_essential:
        print(f"Warning: Missing essential columns: {missing_essential}")
        if len(df.columns) >= 3:
            if 'headline' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'headline'})
            if 'category' not in df.columns:
                df = df.rename(columns={df.columns[1]: 'category'})
            if 'short_description' not in df.columns:
                df = df.rename(columns={df.columns[2]: 'short_description'})
    
    if 'headline' in df.columns:
        df['headline'] = df['headline'].fillna('')
    else:
        df['headline'] = ''
    
    if 'short_description' in df.columns:
        df['short_description'] = df['short_description'].fillna('')
    else:
        df['short_description'] = ''
    
    if 'authors' in df.columns:
        df['authors'] = df['authors'].fillna('')
    else:
        df['authors'] = ''
    
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('UNKNOWN')
    else:
        print("Error: No category column found!")
        return pd.DataFrame()
    
    df['category'] = df['category'].astype(str).str.strip().str.upper()
    df = df[df['category'] != '']
    df = df[df['category'] != 'UNKNOWN']
    
    mask_both_empty = (df['headline'].str.strip() == '') & (df['short_description'].str.strip() == '')
    df = df[~mask_both_empty]
    
    text_parts = []
    if 'headline' in df.columns:
        text_parts.append(df['headline'].str.strip())
    if 'short_description' in df.columns:
        text_parts.append(df['short_description'].str.strip())
    if 'authors' in df.columns:
        text_parts.append(df['authors'].str.strip())
    
    if text_parts:
        df['combined_text'] = text_parts[0]
        for part in text_parts[1:]:
            df['combined_text'] = df['combined_text'] + ' ' + part
    else:
        df['combined_text'] = ''
    
    df = df[df['combined_text'].str.strip() != '']
    
    category_counts = df['category'].value_counts()
    print(f"Total unique categories before filtering: {len(category_counts)}")
    
    min_samples = 10
    valid_categories = category_counts[category_counts >= min_samples].index
    df = df[df['category'].isin(valid_categories)]
    
    max_categories = 20
    if len(valid_categories) > max_categories:
        top_categories = category_counts.head(max_categories).index
        df = df[df['category'].isin(top_categories)]
    
    print(f"Categories after cleaning: {len(df['category'].unique())}")
    print(f"Final dataset size: {len(df)} articles")
    
    return df

# Load dataset
def load_dataset():
    print("Loading dataset...")
    
    possible_files = ['News_Category_Dataset_v3.json', 'news_category.json', 'News_Category.csv']
    file_found = None
    
    for file in possible_files:
        if os.path.exists(file):
            file_found = file
            break
    
    if not file_found:
        print("Error: Dataset file not found.")
        return None
    
    print(f"Found file: {file_found}")
    
    try:
        if file_found.endswith('.json'):
            data = []
            with open(file_found, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            df = pd.DataFrame(data)
        elif file_found.endswith('.csv'):
            df = pd.read_csv(file_found)
        else:
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    df = clean_dataset(df)
    
    if len(df) == 0:
        return None
    
    return df

# Train models
def train_models(df):
    global preprocessor
    print("Training models...")
    
    preprocessor = TextPreprocessor()
    df['processed_text'] = df['combined_text'].apply(preprocessor.preprocess)
    df = df[df['processed_text'].str.strip() != '']
    
    X = df['processed_text']
    y = df['category']
    
    category_counts = y.value_counts()
    min_samples_per_category = category_counts.min()
    
    if min_samples_per_category < 2:
        stratify_param = None
    else:
        stratify_param = y
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_vec, y_train)
    nb_pred = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    
    try:
        lr_model = LogisticRegression(max_iter=500, random_state=42, C=1.0, solver='lbfgs')
        lr_model.fit(X_train_vec, y_train)
        lr_pred = lr_model.predict(X_test_vec)
        lr_acc = accuracy_score(y_test, lr_pred)
        print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    except Exception:
        lr_acc = 0
        lr_model = None
    
    if lr_model is not None and lr_acc > nb_acc:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_acc = lr_acc
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        best_acc = nb_acc
    
    print(f"Best Model: {best_name} (Accuracy: {best_acc:.4f})")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('model_info.pkl', 'wb') as f:
        pickle.dump({'name': best_name, 'accuracy': best_acc}, f)
    
    return vectorizer, preprocessor, best_model, best_name, best_acc

# Desktop Popup Application
class NewsClassifierPopup:
    def __init__(self, root):
        self.root = root
        self.root.title("News Classifier - Always On Top")
        self.root.geometry("600x700")
        
        # Make window always on top
        self.root.attributes('-topmost', True)
        
        # Make window stay on top even when other apps are focused
        self.root.wm_attributes("-topmost", 1)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#667eea", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🔍 News Category Classifier",
            font=("Arial", 18, "bold"),
            bg="#667eea",
            fg="white"
        )
        title_label.pack(pady=10)
        
        info_label = tk.Label(
            header_frame,
            text=f"Model: {model_name} | Accuracy: {round(model_accuracy*100, 2)}%",
            font=("Arial", 10),
            bg="#667eea",
            fg="white"
        )
        info_label.pack()
        
        # Instructions Frame
        inst_frame = tk.Frame(self.root, bg="#f0f0f0")
        inst_frame.pack(fill=tk.X, padx=10, pady=10)
        
        inst_label = tk.Label(
            inst_frame,
            text="📋 Copy any news text from the internet, then click 'Paste & Classify' or press Ctrl+V",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#333",
            wraplength=560,
            justify=tk.LEFT
        )
        inst_label.pack(pady=5)
        
        # Input Frame
        input_frame = tk.Frame(self.root, bg="white")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(
            input_frame,
            text="News Text:",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        # Text input with scrollbar
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            height=10,
            bg="#fafafa",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button Frame
        button_frame = tk.Frame(self.root, bg="white")
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Paste & Classify Button
        paste_btn = tk.Button(
            button_frame,
            text="📋 Paste & Classify",
            font=("Arial", 12, "bold"),
            bg="#667eea",
            fg="white",
            activebackground="#5568d3",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.paste_and_classify
        )
        paste_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Classify Button
        classify_btn = tk.Button(
            button_frame,
            text="🚀 Classify",
            font=("Arial", 12, "bold"),
            bg="#764ba2",
            fg="white",
            activebackground="#6a3d91",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.classify_text
        )
        classify_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Clear Button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=("Arial", 12, "bold"),
            bg="#95a5a6",
            fg="white",
            activebackground="#7f8c8d",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.clear_all
        )
        clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Result Frame
        result_frame = tk.Frame(self.root, bg="white")
        result_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        tk.Label(
            result_frame,
            text="Classification Result:",
            font=("Arial", 11, "bold"),
            bg="white"
        ).pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        # Result display
        self.result_frame_inner = tk.Frame(result_frame, bg="white")
        self.result_frame_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_label = tk.Label(
            self.result_frame_inner,
            text="Paste or type news text above and click 'Classify'",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666",
            wraplength=560,
            justify=tk.CENTER,
            pady=20
        )
        self.result_label.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#ecf0f1", height=30)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(
            footer_frame,
            text="💡 This window stays on top of all apps | Press ESC to minimize",
            font=("Arial", 8),
            bg="#ecf0f1",
            fg="#666"
        )
        footer_label.pack(pady=5)
        
        # Keyboard shortcuts
        self.root.bind('<Control-v>', lambda e: self.paste_and_classify())
        self.root.bind('<Return>', lambda e: self.classify_text())
        self.root.bind('<Escape>', lambda e: self.root.iconify())
        
    def paste_and_classify(self):
        try:
            clipboard_text = pyperclip.paste()
            if clipboard_text:
                self.text_input.delete(1.0, tk.END)
                self.text_input.insert(1.0, clipboard_text)
                self.classify_text()
            else:
                messagebox.showwarning("Empty Clipboard", "No text found in clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not paste from clipboard: {str(e)}")
    
    def classify_text(self):
        text = self.text_input.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Empty Input", "Please enter some news text to classify!")
            return
        
        if len(text) < 10:
            messagebox.showwarning("Text Too Short", "Please enter more text for accurate classification!")
            return
        
        try:
            # Preprocess and classify
            processed = preprocessor.preprocess(text)
            
            if not processed or len(processed) < 5:
                messagebox.showwarning("Invalid Text", "The text doesn't contain enough meaningful words!")
                return
            
            vectorized = vectorizer.transform([processed])
            prediction = model.predict(vectorized)[0]
            confidence_scores = model.predict_proba(vectorized)[0]
            confidence = np.max(confidence_scores) * 100
            
            # Get top 3 predictions
            top_indices = np.argsort(confidence_scores)[-3:][::-1]
            top_categories = model.classes_[top_indices]
            top_confidences = confidence_scores[top_indices] * 100
            
            # Display result
            self.display_result(prediction, confidence, top_categories, top_confidences)
            
        except Exception as e:
            messagebox.showerror("Classification Error", f"Error: {str(e)}")
    
    def display_result(self, category, confidence, top_categories, top_confidences):
        # Clear previous result
        for widget in self.result_frame_inner.winfo_children():
            widget.destroy()
        
        # Main result
        main_frame = tk.Frame(self.result_frame_inner, bg="#667eea", relief=tk.RAISED, borderwidth=2)
        main_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            main_frame,
            text="📰 PREDICTED CATEGORY",
            font=("Arial", 10, "bold"),
            bg="#667eea",
            fg="white"
        ).pack(pady=(10, 5))
        
        tk.Label(
            main_frame,
            text=category,
            font=("Arial", 20, "bold"),
            bg="#667eea",
            fg="white"
        ).pack(pady=5)
        
        # Confidence indicator
        if confidence > 80:
            conf_color = "#4caf50"
            conf_text = "High Confidence"
        elif confidence > 60:
            conf_color = "#ff9800"
            conf_text = "Medium Confidence"
        else:
            conf_color = "#f44336"
            conf_text = "Low Confidence"
        
        conf_frame = tk.Frame(main_frame, bg=conf_color)
        conf_frame.pack(fill=tk.X, pady=(5, 10))
        
        tk.Label(
            conf_frame,
            text=f"{conf_text}: {confidence:.2f}%",
            font=("Arial", 12, "bold"),
            bg=conf_color,
            fg="white"
        ).pack(pady=5)
        
        # Top 3 predictions
        top3_frame = tk.Frame(self.result_frame_inner, bg="white")
        top3_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            top3_frame,
            text="Top 3 Predictions:",
            font=("Arial", 10, "bold"),
            bg="white"
        ).pack(anchor=tk.W, pady=(5, 5))
        
        for i, (cat, conf) in enumerate(zip(top_categories, top_confidences), 1):
            pred_frame = tk.Frame(top3_frame, bg="#f0f0f0", relief=tk.SOLID, borderwidth=1)
            pred_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(
                pred_frame,
                text=f"{i}. {cat}",
                font=("Arial", 10, "bold"),
                bg="#f0f0f0",
                anchor=tk.W
            ).pack(side=tk.LEFT, padx=10, pady=5)
            
            tk.Label(
                pred_frame,
                text=f"{conf:.2f}%",
                font=("Arial", 10),
                bg="#f0f0f0",
                fg="#667eea",
                anchor=tk.E
            ).pack(side=tk.RIGHT, padx=10, pady=5)
    
    def clear_all(self):
        self.text_input.delete(1.0, tk.END)
        
        for widget in self.result_frame_inner.winfo_children():
            widget.destroy()
        
        self.result_label = tk.Label(
            self.result_frame_inner,
            text="Paste or type news text above and click 'Classify'",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666",
            wraplength=560,
            justify=tk.CENTER,
            pady=20
        )
        self.result_label.pack(fill=tk.BOTH, expand=True)

# Main execution
def main():
    global model, vectorizer, preprocessor, model_name, model_accuracy
    
    print("="*60)
    print("NEWS CLASSIFICATION SYSTEM - DESKTOP POPUP")
    print("="*60)
    
    # Load dataset
    df = load_dataset()
    
    if df is None or len(df) == 0:
        print("Error: Could not load dataset.")
        return
    
    # Train or load models
    try:
        if not all(os.path.exists(f) for f in ['model.pkl', 'vectorizer.pkl', 'preprocessor.pkl']):
            vectorizer, preprocessor, model, model_name, model_accuracy = train_models(df)
        else:
            print("Loading trained models...")
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('model_info.pkl', 'rb') as f:
                info = pickle.load(f)
                model_name = info['name']
                model_accuracy = info['accuracy']
    except Exception as e:
        print(f"Error: {e}")
        vectorizer, preprocessor, model, model_name, model_accuracy = train_models(df)
    
    print("="*60)
    print("Launching Desktop Popup Application...")
    print("Window will stay on top of all applications")
    print("="*60)
    
    # Launch GUI
    root = tk.Tk()
    app = NewsClassifierPopup(root)
    root.mainloop()

if __name__ == "__main__":
    main()