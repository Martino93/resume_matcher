import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

'''
TF-IDF (Term Frequency-inverse Document Frequency) is a feature that associates a number with each word in a document to represent its relevance. TF-IDF has two parts: 

Term Frequency (TF): Indicates the frequency of each word in the document or dataset.

Inverse Document Frequency (IDF): Tells us how important the word is to the document.

Cosine similarity is a metric that measures the similarity between two vectors. It's often used in text analysis to measure document similarity.
'''


def extract_text_from_pdf(file):
    # with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Ensure text is successfully extracted
            text += page_text
    return text

def process_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(filtered_words)

def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

# Paths to your resume and job description PDFs
# resume_path = 'resume.pdf'
# job_desc_path = 'jobdesc2.pdf'

# # Extract and process text
# resume_text = process_text(extract_text_from_pdf(resume_path))
# job_desc_text = process_text(extract_text_from_pdf(job_desc_path))

# # Calculate similarity score
# similarity_score = calculate_similarity(resume_text, job_desc_text)
# print(f"Similarity Score: {similarity_score:.2f}")
