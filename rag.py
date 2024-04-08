import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os

HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def get_response(url, query):
    if not HF_TOKEN:
        raise ValueError("Hugging Face API token not found in environment variable HUGGINGFACEHUB_API_TOKEN")

    # Cache data to improve responsiveness
    cache = {}

    # Load data from URL
    if url not in cache:
        data = WebBaseLoader(url)
        cache[url] = data.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    chunking = text_splitter.split_documents(cache[url])

    # Create embeddings
    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    # Create vector store
    vectors = Chroma.from_documents(chunking, embedding)

    # Create retriever
    retriever = vectors.as_retriever(search_type="mmr", search_kwargs={"k": 1})

    # Load open-source LLM
    model = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-alpha",
        model_kwargs={"temperature": 0.8, "max_new_tokens": 512, "max_length": 64},
    )

    qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")
    template = "{query}"
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Attempt to retrieve answer from website
    website_response = rag_chain.invoke(query)
    if website_response and 'AI:' in website_response:
        return website_response.split('AI:')[1].strip()

    # If website response is not available, use the model's response
    model_response = qa.run(query)
    if model_response:
        return model_response.strip()

    # If no relevant information is found, provide a generic answer
    return "I'm sorry, I couldn't find any relevant information about that."

def main():
    st.title("RAG System")
    url = st.text_input("Enter the URL:")
    query = st.text_input("Enter your query:")
    if url and query:
        response = get_response(url, query)
        assistant_response = response.split('Human:')[1].strip() if 'Human:' in response else response
        st.write(assistant_response)

if __name__ == "__main__":
    main()