import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import streamlit as st
import pickle

load_dotenv()

GoogleAPIKEY = os.getenv("Google_API_KEY")

genai.configure(api_key=GoogleAPIKEY)

faiss_index_dir = "faiss_index"
faiss_index_path = os.path.join(faiss_index_dir, "faiss_index.pkl")

os.makedirs(faiss_index_dir, exist_ok=True)

def getpdftext(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text =text+page.extract_text()
    return text

def texttochunk(text, chunk_size=10000, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def getconversation(prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'questions'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def storetoFAISS(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="embedding-v1")
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        with open(faiss_index_path, "wb") as f:
            pickle.dump(vectorstore, f)
    except Exception as e:
        raise

def userinput(userquestion, chain):
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-v1")
    try:
        with open(faiss_index_path, "rb") as f:
            vectorstore = pickle.load(f)
        docs = vectorstore.similarity_search(userquestion)
        response = chain({"input_documents": docs, "question": userquestion}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except FileNotFoundError as e:
        st.error("FAISS index not found. Please upload a PDF file first.")

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":book:")
    st.header("Chat using Google's Gemini")

    prompt_template = "Given the following context: {context}\nAnswer the following questions: {questions}"
    chain = getconversation(prompt_template)

    with st.sidebar:
        st.title("Menu")
        st.write("Upload your PDF file")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            try:
                with st.spinner("Please wait..."):
                    rawtext = getpdftext(uploaded_file)
                    text_chunks = texttochunk(rawtext, chunk_size=400, chunk_overlap=40)
                    storetoFAISS(text_chunks)
                    st.success("Completion")
            except Exception as e:
                st.error(f"Error occurred in uploading: {e}")

    user_question = st.text_input("Enter your question:")
    if user_question:
        userinput(user_question, chain)

if __name__ == "__main__":
    main()
