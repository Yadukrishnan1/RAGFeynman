import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login
from getpass import getpass
import streamlit as st


# Assuming you have a text file with your document
loader = TextLoader("../data/all_snippets.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize the embedding function
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)

# Save the index to disk
vectorstore.save_local("faiss_index")

print("FAISS vector store created successfully!")

# Get YOUR HF tokens
def get_huggingface_token():
    # Token from environment variables
    token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    
    # If not found, prompt the user
    if not token:
        print("Hugging Face token not found in environment variables.")
        token = getpass("Please enter your Hugging Face token: ")
        
        # Save the token as an environment variable for this session
        os.environ['HUGGINGFACE_TOKEN'] = token
    
    return token

# Loading models (I used TinyLlama, Gemma and meta-llama)
def load_model(model_choice):
    if model_choice == "tinyllama":
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif model_choice == "gemma":
        model_name = "google/gemma-2b"
    elif model_choice == "llama":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        raise ValueError("Invalid model choice. Choose 'tinyllama', 'gemma', or 'llama'.")

    # We need to login to Hugging Face if using Gemma or Llama as they're gated mdoels
    if model_choice in ["gemma", "llama"]:
        token = get_huggingface_token()
        login(token=token)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    return tokenizer, model

# Choose your model here: "tinyllama", "gemma", or "llama"
model_choice = "tinyllama"

# Load the chosen model and tokenizer
tokenizer, model = load_model(model_choice)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def retrieve_context(query, k=3):
    return vectorstore.similarity_search(query, k=k)

def generate_response(query, context):
    # Construct the prompt
    prompt = f"Context: {context}\n\nHuman: {query}\n\nAssistant:"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response #response.split("Assistant:")[-1].strip()

# Main RAG function
def rag_query(query):
    # Retrieve relevant contexts
    contexts = retrieve_context(query)
    context_text = "\n".join([doc.page_content for doc in contexts])

    # Construct prompt and generate response
    response = generate_response(query, context_text)

    return response.split("Assistant:")[-1].strip()

