import os
import time
import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(page_title="SamacharAI", page_icon="📰", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f1a;
        color: #e2e8f0;
    }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }
    .card {
        background: #131929;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .answer-card {
        background: #0f1e35;
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        font-size: 1.05rem;
        line-height: 1.75;
        color: #e2e8f0;
    }
    .hindi-card {
        background: #1a0f35;
        border-left: 4px solid #f472b6;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        font-size: 1.1rem;
        line-height: 2;
        color: #f1e9ff;
    }
    .source-pill {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.35rem 0.8rem;
        font-size: 0.8rem;
        color: #94a3b8;
        margin: 0.3rem 0.3rem 0 0;
        word-break: break-all;
    }
    .lang-tag {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
        display: inline-block;
    }
    .lang-en { background: #0c2d48; color: #38bdf8; }
    .lang-hi { background: #2d0c2d; color: #f472b6; }

    [data-testid="stSidebar"] {
        background: #0d1220 !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    .sidebar-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8 !important;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        background: #131929 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 0.7rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
        color: #0b0f1a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        width: 100%;
    }
    .progress-text {
        color: #38bdf8;
        font-size: 0.92rem;
        font-weight: 500;
        padding: 0.8rem 1rem;
        background: #0c1e30;
        border-radius: 8px;
        border: 1px solid #1e3a52;
        margin: 0.5rem 0;
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1.5rem 0;
    }

    /* Sidebar toggle button — always visible */
    [data-testid="collapsedControl"],
    section[data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebarCollapsedControl"] button,
    div[data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #38bdf8 !important;
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="sidebar-title">SamacharAI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Paste News Article URLs**")
    st.caption("Supports Hindi, English, or any language")

    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}", placeholder=f"https://example.com/news-{i+1}", key=f"url_{i}")
        urls.append(url)

    st.markdown("---")
    process_url_clicked = st.button("Process URLs")

    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption("LLM: llama3.2:3b (Ollama)")
    st.caption("Embeddings: nomic-embed-text")
    st.caption("Vector DB: FAISS")
    st.caption("100% Free — Runs Locally")


st.markdown('<div class="hero-title">SamacharAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ask questions from any news article — get answers in English & Hindi instantly.</div>', unsafe_allow_html=True)

store_path = "faiss_store_samachar"


if process_url_clicked:
    valid_urls = [u for u in urls if u.strip()]

    if not valid_urls:
        st.warning("Please enter at least one URL in the sidebar.")
    else:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            progress = st.empty()

            progress.markdown('<div class="progress-text">Step 1a — Loading articles from URLs...</div>', unsafe_allow_html=True)
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()

            progress.markdown('<div class="progress-text">Step 1b — Splitting text into chunks...</div>', unsafe_allow_html=True)
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(data)

            progress.markdown('<div class="progress-text">Step 1c & 1d — Generating embeddings and saving to FAISS...</div>', unsafe_allow_html=True)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.from_documents(chunks, embeddings)
            time.sleep(1)

            vector_store.save_local(store_path)

            progress.markdown('<div class="progress-text">Done! Ask your question below.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### Ask a Question")
st.caption("Type in English or Hindi — you will get answers in both languages.")
query = st.text_input("", placeholder="e.g. What is the main news? / मुख्य समाचार क्या है?", label_visibility="collapsed")


if query:
    if not os.path.exists(store_path):
        st.warning("Please process URLs first using the sidebar!")
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        english_prompt = PromptTemplate(
            template="""
You are a helpful news research assistant.
Answer ONLY from the provided context.
If the context is insufficient, say you don't know.
Be concise and clear.

Context:
{context}

Question: {question}

Answer in English:
            """,
            input_variables=["context", "question"]
        )

        hindi_prompt = PromptTemplate(
            template="""
You are a helpful news research assistant.
Answer ONLY from the provided context.
If the context is insufficient, say you don't know.
You MUST reply in Hindi using Devanagari script only.

Context:
{context}

Question: {question}

हिंदी में उत्तर दें:
            """,
            input_variables=["context", "question"]
        )

        llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
        parser = StrOutputParser()

        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        english_chain = parallel_chain | english_prompt | llm | parser
        hindi_chain   = parallel_chain | hindi_prompt   | llm | parser

        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Generating English answer..."):
                english_answer = english_chain.invoke(query)
            st.markdown('<span class="lang-tag lang-en">English</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{english_answer}</div>', unsafe_allow_html=True)

        with col2:
            with st.spinner("Generating Hindi answer..."):
                hindi_answer = hindi_chain.invoke(query)
            st.markdown('<span class="lang-tag lang-hi">Hindi</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="hindi-card">{hindi_answer}</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        retrieved_docs = retriever.invoke(query)
        sources = list({doc.metadata.get("source", "") for doc in retrieved_docs if doc.metadata.get("source")})
        if sources:
            st.markdown("**Sources**")
            pills = "".join([f'<span class="source-pill">🔗 {s}</span>' for s in sources])
            st.markdown(pills, unsafe_allow_html=True)