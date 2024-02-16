# Imports
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  
GOOGLE_API_KEY ='AIzaSyDkbx_T5C3cTyaPNpYxE8GeJLknug6vnAI'
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure Google Generative AI

# Extracts text from all pages of provided PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Splits text into chunks of 10,000 characters with 1,000 character overlap
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Creates and saves a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Creates and returns a conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
Answer the question concisely, focusing on the most relevant and important details from the PDF context. 
Refrain from mentioning any mathematical equations, even if they are present in provided context. 
Focus on the textual information available. Please provide direct quotations or references from PDF
to back up your response. If the answer is not found within the PDF, 
please state "answer is not available in the context."\n\n

Context:\n {context}?\n
Question: \n{question}\n

Example response format:

Overview: 
(brief summary or introduction)

Key points: 
(point 1: paragraph for key details)
(point 2: paragraph for key details)
...

Use a mix of paragraphs and points to effectively convey the information.
"""

# Adjust temperature parameter to lower value to: 
# reduce model creativity & focus on factual accuracy
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Processes user question and provides a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"],"")  

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="")
    st.header("Chat with multiple PDFs using AI 💬")

    user_question = st.text_input("Ask a Question from PDF file(s)")  

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu ✨")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button ",
                                     accept_multiple_files=True)

        if st.button("Submit & Process"): 
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ✨")

if __name__ == "__main__":
    main()