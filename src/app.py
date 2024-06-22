import os
import traceback
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
import PyPDF2
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Define function to create vector store from URL
def get_vectorstore_from_url(url, max_depth, pdf_text=None):
    try:
        if not os.path.exists('src/chroma'):
            os.makedirs('src/chroma')
        if not os.path.exists('src/scrape'):
            os.makedirs('src/scrape')

        documents = []
        if url:
            urls = scrape_urls(url, max_depth)
            loader = WebBaseLoader(urls)
            documents = loader.load()

        if pdf_text:
            documents.append(Document(page_content=pdf_text, metadata={"source": "uploaded_pdf"}))

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
        document_chunks = text_splitter.split_documents(documents)

        # Create embeddings
        embedding = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vector_store = Chroma.from_documents(document_chunks, embedding)
        return vector_store, len(documents)
    except Exception as e:
        st.error(f"Error occurred during processing: {e}")
        traceback.print_exc()
        return None, 0

# Define function to create context retriever chain
def get_context_retriever_chain(vector_store):
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

# Define function to create conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-alpha",
        model_kwargs={"temperature": 0.8, "max_new_tokens": 512, "max_length": 64},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Provide a concise and relevant answer to the user's question based on the given context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Context: {context}"),
        ("human", "Answer:")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    conversation_rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    return conversation_rag_chain

# Define function to get response
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    # Extract only the AI's response
    ai_response = response['answer'].split("Answer:", 1)[-1].strip()
    return ai_response

# Streamlit app configuration
st.set_page_config(page_title="WEB-AI 🤖: Chat With Websites", page_icon="🤖")
st.title("WEB-AI 🤖: Your Web Assistant")

# Initialize session state
if "freeze" not in st.session_state:
    st.session_state.freeze = False
if "max_depth" not in st.session_state:
    st.session_state.max_depth = 1
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "len_urls" not in st.session_state:
    st.session_state.len_urls = 0

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() if page.extract_text() else ""
    return text

# Sidebar configuration
# Sidebar configuration
with st.sidebar:
    st.header("WEB-AI 🤖")
    website_url = st.text_input("Website URL (optional)")

    st.title("PDF Uploader")
    uploaded_file = st.file_uploader("Upload a PDF file (optional)", type=["pdf"])

    st.session_state.max_depth = st.slider("Select maximum scraping depth:", 1, 5, 1, disabled=st.session_state.freeze)
    if st.button("Proceed", disabled=st.session_state.freeze):
        st.session_state.freeze = True

    pdf_text = None
    if uploaded_file is not None:
        st.write(f"Uploaded file: {uploaded_file.name}")
        pdf_text = extract_text_from_pdf(uploaded_file)

# Main app logic
if website_url or pdf_text:
    if st.session_state.freeze:
        if st.session_state.vector_store is None:
            with st.spinner("Processing content..."):
                st.session_state.vector_store, st.session_state.len_docs = get_vectorstore_from_url(website_url, st.session_state.max_depth, pdf_text)
                if st.session_state.vector_store:
                    st.success(f"Processing completed! Total Documents: {st.session_state.len_docs}")
                else:
                    st.error("Failed to create vector store.")
                    st.session_state.freeze = False
        else:
            st.sidebar.success(f"Total Documents Processed: {st.session_state.len_docs}")

        user_query = st.chat_input("Type your message here...")
        if user_query:
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

# Sidebar footer
with st.sidebar:
    st.markdown('---')
    st.markdown('Connect with me:')
    st.markdown('[LinkedIn](https://www.linkedin.com/in/sunil-giri77/)')
    st.markdown('[GitHub](https://github.com/sunilgiri7)')
    st.markdown('[Email](mailto:seungiri841@gmail.com)')
