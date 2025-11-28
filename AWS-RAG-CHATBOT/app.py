import json
import os
import sys
import boto3
import base64
import streamlit as st


from langchain_classic.embeddings import BedrockEmbeddings
from langchain_classic.llms.bedrock import Bedrock
from langchain_classic.chains import RetrievalQA


import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


try:
    bedrock = boto3.client(service_name="bedrock-runtime")
    bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)
except Exception as e:
    st.error(f"Error initializing AWS Bedrock client or embedding model. Ensure credentials and region are configured. Error: {e}")
    bedrock = None
    bedrock_embedding = None



def data_ingestion():
    """Loads documents from the 'data' directory and splits them into chunks."""
    try:
        if not os.path.exists("data"):
            st.error("The 'data' directory does not exist. Please create it and place your PDFs inside.")
            return []
            
        loader = PyPDFDirectoryLoader("data")
        document = loader.load()

        if not document:
             st.warning("No documents found in the 'data' directory.")
             return []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(document)
        return docs
    except Exception as e:
        st.error(f"Error during data ingestion: {e}")
        return []

def get_vector_store(docs):
    """Creates a FAISS vector store from documents and saves it locally."""
    if not docs or bedrock_embedding is None:
        return
        
    st.info(f"Processing {len(docs)} document chunks...")
    try:
        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embedding
        )
        vectorstore_faiss.save_local("faiss_index")
        st.success("FAISS index saved successfully as 'faiss_index' directory.")
    except Exception as e:
        st.error(f"Error creating/saving vector store: {e}")



def get_llm():
    """Initializes and returns the Titan LLM model with corrected parameters."""
    if bedrock is None:
        return None
        

    llm = Bedrock(
    model_id="meta.llama3-8b-instruct-v1:0",
    client=bedrock,
    model_kwargs={"max_gen_len": 512}
)
    return llm

prompt_template = PromptTemplate(
    template="""
Human: Use the following context to answer the question. 
First, provide a direct answer. Then, provide a detailed explanation of the topic, aiming for around 250 words. 
If the context does not contain the answer, state that you do not know. Do not try to make up the answer.
Context: {context}
Question: {question}
Assistant:
""",
    input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    """Creates and runs the RetrievalQA chain."""
    if llm is None or vectorstore_faiss is None:
        return "LLM or Vector Store not initialized."
        
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        answer = qa({"query": query})
        return answer['result']
    except Exception as e:
        st.error(f"An error occurred during query execution (Bedrock API or LangChain): {e}")
        return f"An error occurred during query execution: {e}"

def main():
    st.set_page_config(page_title="chataws", layout="wide")
    st.header("RAG USING AWS BEDROCK")

    faiss_exists = os.path.exists("faiss_index")

    with st.sidebar:
        st.title("Update or create your vectorstore")
        
        if st.button("Update Vectors"):
            with st.spinner("Suffer While We Work..."):
                docs = data_ingestion() 
                if docs:
                    get_vector_store(docs)
                    st.success("Affirmative👍")
                else:
                    st.warning("Vector store update failed. Check error messages above.")
        
        if faiss_exists:
            st.markdown(f"**Index Status:** ✅ Ready to use.")
        else:
            st.markdown(f"**Index Status:** 🛑 Missing. Please click 'Update Vectors'.")

    user_question = st.text_input("Ask anything. Don't feel shy 🪖", key="user_input")

    if st.button("Titan Output"):
        if not user_question:
            st.warning("Please enter a question.")
            return

        if not faiss_exists:
            st.error("Vector store index not found. Please click 'Update Vectors' first.")
            return

        with st.spinner("Suffer while we work..."):
            try:
                faiss_index = FAISS.load_local(
                    'faiss_index', 
                    bedrock_embedding, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                st.error(f"Failed to load FAISS index. Error: {e}")
                return
            
            llm = get_llm()
            if llm is None:
                return

            response = get_response_llm(llm, faiss_index, user_question)
            
            st.subheader("Assistant Response:")
            st.markdown(response)
            
            st.success("Affirmative👍")


if __name__ == "__main__":
    main()