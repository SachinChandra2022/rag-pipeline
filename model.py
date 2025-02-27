import PyPDF2 as pdf
import numpy as np
import nltk
import string
import random
import os
import json
import google.generativeai as genai
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini API
genai.configure(api_key="AIzaSyAB6z1s_iN4R7tlpeRJJniQ4o2yIax-sFg")
model = genai.GenerativeModel('gemini-pro')

def download_nltk_data():
    try:
        # Download all required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Set NLTK data path
        nltk.data.path.append('./nltk_data')
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
        print("Please run these commands manually:")
        print("python -m nltk.download('punkt')")
        print("python -m nltk.download('punkt_tab')")
        print("python -m nltk.download('wordnet')")
        print("python -m nltk.download('omw-1.4')")

# Call the function immediately
download_nltk_data()

def combine_pdfs(pdf_files):
    combined_text = ""
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = pdf.PdfReader(file)
                for page in pdf_reader.pages:
                    combined_text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    return combined_text

def get_relevant_context(query: str, context: str, max_tokens: int = 1000) -> str:
    """Extract most relevant parts of the context for the query using TF-IDF"""
    sentences = nltk.sent_tokenize(context)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([query] + sentences)
    
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    ranked_sentences = [(score, sent) for score, sent in zip(similarity_scores, sentences)]
    ranked_sentences.sort(reverse=True)
    
    relevant_context = ""
    current_length = 0
    
    for _, sentence in ranked_sentences:
        if current_length + len(sentence) > max_tokens:
            break
        relevant_context += sentence + " "
        current_length += len(sentence)
    
    return relevant_context.strip()

def get_gemini_response(query: str, context: str) -> str:
    """Get response from Gemini using the query and context"""
    prompt = f"""You are NidhiPath, a helpful financial advisor assistant. Use the following context to answer the user's question.
    If you cannot find the answer in the context, say so politely and suggest related topics you can help with.
    
    Context: {context}
    
    User Question: {query}
    
    Please provide a clear, concise, and accurate response based on the context provided. Include specific details when available.
    If you make any financial recommendations, include appropriate disclaimers."""

    try:
        response = model.generate_content(
            contents=prompt
        )
        if response and hasattr(response, 'text'):
            return response.text
        return "I apologize, but I couldn't generate a response at this time."
    except Exception as e:
        print(f"Debug - Error details: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def nidhi_path(sentence: str) -> str:
    # List of financial knowledge PDFs
    pdf_files = [
        "RBI_guidelines-home-loan.pdf"
    ]
    
    try:
        # Check if combined text file exists, if not create it
        if not os.path.exists('combined_financial_text.txt'):
            combined_text = combine_pdfs(pdf_files)
            if not combined_text:
                return "Error: No content could be extracted from PDF files."
            with open('combined_financial_text.txt', 'w', encoding='utf-8') as output_file:
                output_file.write(combined_text)

        # Read the combined text
        with open('combined_financial_text.txt', 'r', encoding='utf-8', errors='ignore') as f:
            context = f.read()
            if not context:
                return "Error: No content available in the knowledge base."

    except Exception as e:
        print(f"Error in NidhiPath function: {str(e)}")
        return "I'm sorry, but I encountered an error. Please try again later."

    # Handle greetings
    greet_inputs = ('hello', 'hi', 'hey', 'greetings')
    if sentence.lower() in greet_inputs:
        return "Hello! I'm NidhiPath, your financial assistant. How can I help you today?"

    if sentence.lower() in ['bye', 'goodbye', 'exit']:
        return "Goodbye! Feel free to return if you have more financial questions."

    if sentence.lower() in ['thank you', 'thanks']:
        return "You're welcome! Is there anything else you'd like to know about?"

    # Get relevant context for the query
    relevant_context = get_relevant_context(sentence, context)
    
    # Get response from Gemini
    response = get_gemini_response(sentence, relevant_context)
    
    return response

# For testing purposes
if __name__ == "__main__":
    output = nidhi_path("hi")
    print(output)














