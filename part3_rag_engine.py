import logging
from typing import Tuple, List
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Part3_RAG_Compliance_Engine")

st.set_page_config(page_title="Part 3: RAG Credit Compliance QA Engine", layout="wide")
st.title("Part III: Retrieval-Augmented Generation (RAG) Credit Compliance Engine")
st.markdown("---")


def validate_rag_inputs(api_key: str, doc_text: str, query_text: str, use_mock: bool = False) -> Tuple[bool, str]:
    if not doc_text or len(doc_text.strip()) < 30:
        return False, "Policy knowledge base text is too short (minimum 30 characters required)."
    if not query_text or len(query_text.strip()) < 5:
        return False, "Compliance query is too short (minimum 5 characters required)."
    if use_mock:
        return True, ""
    if not api_key or not api_key.strip():
        return False, "OpenAI API Key is required. Enter it in the sidebar or enable Mock Mode."
    if not api_key.strip().startswith("sk-"):
        return False, "Invalid API Key format. OpenAI API keys must begin with 'sk-'."
    return True, ""


st.sidebar.header("Configuration")
use_mock_mode = st.sidebar.checkbox("Enable Mock Mode (Offline Testing)", value=False)
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter key starting with sk-")
st.sidebar.markdown("---")
st.sidebar.caption("Powered by LangChain, FAISS Vector Store & OpenAI GPT-4o-mini")

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("1. Bank Credit Policy Knowledge Base")
    default_doc = (
        "BANK COMMERCIAL LENDING & CREDIT RISK POLICY GUIDELINES (REVISED Q3)\n"
        "1. MAXIMUM LEVERAGE COVENANT: The maximum permissible Debt-to-EBITDA ratio for tier-1 corporate borrowers is set at 3.0x. "
        "Any ratio exceeding 3.5x triggers an immediate mandatory Credit Watchlist Grade B classification and requires a formal remediation audit.\n"
        "2. LIQUIDITY & CAPITAL ADEQUACY: Borrowers must maintain a minimum unencumbered cash balance equal to 15% of outstanding total net debt. "
        "If liquidity falls below $500M, dividend distributions are strictly prohibited without prior underwriting committee waiver.\n"
        "3. INTEREST COVERAGE RATIO: EBITDA-to-Interest expense must remain above 4.0x at all quarterly reporting intervals. "
        "A decline below 2.5x constitutes an automatic Event of Default under Section 8.2 of the Master Credit Facility Agreement.\n"
        "4. AUDIT & REPORTING COMPLIANCE: Borrowers are required to submit quarterly audited financial disclosures within 45 days of quarter end. "
        "Late submissions incur a penalty fee of 25 bps on total committed facility limits."
    )
    doc_payload = st.text_area("Ingest Corporate Credit Guidelines / Regulatory Manual:", height=240, value=default_doc)

with col_right:
    st.subheader("2. Compliance & Credit Risk Inquiry")
    user_query = st.text_input(
        "Enter Audit Query / Compliance Risk Question:",
        value="What happens if a corporate borrower's Debt-to-EBITDA ratio exceeds 3.5x, and what are the liquidity requirements?"
    )
    chunk_size = st.slider("Text Chunk Size (Characters)", min_value=150, max_value=800, value=350, step=50)
    chunk_overlap = st.slider("Chunk Overlap (Characters)", min_value=0, max_value=150, value=40, step=10)


def execute_mock_rag(doc_text: str) -> Tuple[str, List[str]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)
    chunks = splitter.split_documents([Document(page_content=doc_text)])
    retrieved = [c.page_content for c in chunks[:3]]
    answer = (
        "Based strictly on the ingested Credit Risk Guidelines:\n\n"
        "1. Debt-to-EBITDA Violation (>3.5x): Triggers an immediate mandatory Credit Watchlist Grade B classification "
        "and mandates a formal remediation audit.\n\n"
        "2. Liquidity Requirements: Corporate borrowers must maintain a minimum unencumbered cash balance equal to 15% of total net debt. "
        "If liquidity drops below $500M, dividend distributions are strictly prohibited without underwriting committee waiver."
    )
    return answer, retrieved


def execute_live_rag(doc_text: str, query: str, api_key: str, c_size: int, c_overlap: int) -> Tuple[str, List[str]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    chunks = splitter.split_documents([Document(page_content=doc_text)])
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    retrieved_texts = [d.page_content for d in retrieved_docs]
    context_block = "\n\n".join(retrieved_texts)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert BFSI Credit Risk Auditor. Answer the question using ONLY the provided context.\n"
        "If the answer cannot be determined from the context, state that context is insufficient.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer strictly based on context:"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.1, timeout=30.0)
    response = (prompt | llm).invoke({"context": context_block, "question": query}).content
    return response, retrieved_texts


if st.button("Run RAG Pipeline (Ingest, Retrieve & Answer)", type="primary"):
    valid, msg = validate_rag_inputs(api_key_input, doc_payload, user_query, use_mock=use_mock_mode)
    if not valid:
        logger.warning(f"Validation error: {msg}")
        st.error(f"Validation Error: {msg}")
    else:
        try:
            with st.spinner("Executing FAISS RAG retrieval and grounded synthesis..."):
                if use_mock_mode:
                    ans, chunks = execute_mock_rag(doc_payload.strip())
                else:
                    ans, chunks = execute_live_rag(
                        doc_payload.strip(), user_query.strip(), api_key_input.strip(), chunk_size, chunk_overlap
                    )
                st.markdown("---")
                st.subheader("RAG Synthesized Compliance Response")
                st.success(ans)
                st.subheader("Retrieved Context Chunks (Vector Search Grounding)")
                for idx, ch in enumerate(chunks, 1):
                    with st.expander(f"Retrieved Source Passage {idx}"):
                        st.text(ch)
                logger.info("Part 3 RAG pipeline executed successfully.")
        except Exception as err:
            logger.critical(f"Part 3 execution failure: {str(err)}")
            st.error(f"Execution Error: {str(err)}")

st.markdown("---")
st.caption("Part III: Cloud Native RAG Architecture - BFSI Credit Risk Engine")
