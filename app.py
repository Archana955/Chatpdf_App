import streamlit as st
import os  # Added for loading environment variables
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()

# Sidebar contents
with st.sidebar:
    st.title('💬 Chat App')
    st.markdown(''' 
                
    ## About
    This app is LLM powered chatbot built using:
    - [streamlit](http://streamlit.io/)
    - [Langchain](http://python.langchain.com/)
    
    ''')
    add_vertical_space(10)
    st.write('Made by [Archana Parmar]')

def main():
    st.header('Chat with PDF 🗨️')
    
    # Upload a PDF file
    pdf = st.file_uploader('Upload your PDF', type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len 
        )
        chunks = text_splitter.split_text(text=text)
        
        st.write(chunks)
        
        # Embedding
        embeddings = HuggingFaceEmbeddings()
        
        vectorstores = FAISS.from_texts(chunks, embedding=embeddings)
        user_question = st.text_input("Ask Question about your PDF:")
        if user_question:
            docs = vectorstores.similarity_search(user_question)
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 5, "max_length": 64})
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)

if __name__ == '__main__':
    main()