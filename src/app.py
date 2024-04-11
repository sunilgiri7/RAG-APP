import os
import streamlit as st
from utils.get_urls import scrape_urls
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
import re

load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def get_vectorstore_from_url(url, max_depth):
    try:
        if not os.path.exists('src/chroma'):
            os.makedirs('src/chroma')
        if not os.path.exists('src/scrape'):
            os.makedirs('src/scrape')

        urls = scrape_urls(url, max_depth)
        loader = WebBaseLoader(urls)
        document = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
        document_chunks = text_splitter.split_documents(document)

        # Create embeddings
        embedding = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vector_store = Chroma.from_documents(document_chunks, embedding)
        return vector_store, len(urls)
    except Exception as e:
        st.error(f"Error occurred during scraping: {e}")
        return None, 0


def get_context_retriever_chain(vector_store):
    # Load open-source LLM
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-alpha",
        model_kwargs={"temperature": 0.8, "max_new_tokens": 512, "max_length": 64},
    )
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain):
    # Load open-source LLM
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-alpha",
        model_kwargs={"temperature": 0.8, "max_new_tokens": 512, "max_length": 64},
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that can provide concise and relevant answers based on the given context. Do not output the entire context, only the relevant information to answer the user's query."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Context: {context}"),
        ("user", "Given the above context, provide a concise and relevant answer to the original query.")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    conversation_rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
    return conversation_rag_chain

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": [],
        "input": user_input
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="WebChat 🤖: Chat With Websites", page_icon="🤖")
st.title("WebChat 🤖: Your Web Assistant")

if "freeze" not in st.session_state:
    st.session_state.freeze = False
if "max_depth" not in st.session_state:
    st.session_state.max_depth = 1

# sidebar
with st.sidebar:
    st.header("AI Assistant 🤖")
    website_url = st.text_input("Website URL")
    
    st.session_state.max_depth = st.slider("Select maximum scraping depth:", 1, 5, 1, disabled=st.session_state.freeze)
    if st.button("Proceed", disabled=st.session_state.freeze):
        st.session_state.freeze = True
    
if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    if st.session_state.freeze:
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "vector_store" not in st.session_state:
            with st.sidebar:
                with st.spinner("Scrapping Website..."):
                    st.session_state.vector_store, st.session_state.len_urls = get_vectorstore_from_url(website_url,
                                                                                                        st.session_state.max_depth)
                    st.write(f"Total Pages Scrapped: {st.session_state.len_urls}")
                    st.success("Scraping completed, 🤖 Ready!")

        else:
            with st.sidebar:
                    st.write(f"Total Pages Scrapped: {st.session_state.len_urls}")
                    st.success("🤖 Ready!")

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query:
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    
with st.sidebar:
    st.sidebar.markdown('---')
    st.sidebar.markdown('Connect with me:')
    st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/sunil-giri77/)')
    st.sidebar.markdown('[GitHub](https://github.com/sunilgiri7)')
    st.sidebar.markdown('[Email](mailto:seungiri841@gmail.com)')