import concurrent.futures
import logging
from typing import Tuple
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Configure Logging & System Observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Part1_Financial_NLP_Suite")

st.set_page_config(page_title="Part 1: Multi-Task Financial NLP Suite", layout="wide")
st.title("Part I: Multi-Task Financial Risk & Credit Intelligence Engine")
st.markdown("---")

def validate_inputs(api_key: str, text_content: str, use_mock: bool = False) -> Tuple[bool, str]:
    """Validates API key syntax or verifies mock mode enablement."""
    if use_mock:
        if not text_content or len(text_content.strip()) < 20:
            return False, "Input text payload is too short. Please provide at least 20 characters of financial text."
        return True, ""
    
    if not api_key or not api_key.strip():
        return False, "OpenAI API Key is required. Please enter it in the sidebar or enable Mock Mode."
    if not api_key.strip().startswith("sk-"):
        return False, "Invalid API Key format. OpenAI API keys must begin with 'sk-'."
    if not text_content or len(text_content.strip()) < 20:
        return False, "Input text payload is too short. Please provide at least 20 characters of financial text."
    return True, ""

# Sidebar Configuration
st.sidebar.header("Configuration")
use_mock_mode = st.sidebar.checkbox("Enable Mock Mode (Offline Testing)", value=False, help="Runs local simulation without calling OpenAI API.")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your key starting with sk-")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by OpenAI GPT-4o-mini via LangChain")

input_payload = st.text_area(
    "Enter Corporate Financial Statement / SEC Filing Extract:",
    height=180,
    value="Company XYZ reported Q3 revenue of $4.2B, down 4% YoY. Net debt increased to $1.8B following the acquisition, raising Debt-to-EBITDA to 3.8x. Management lowered full-year EBITDA guidance by 150 bps due to supply chain disruptions and rising interest expenses. Liquidity remains stable with $600M cash."
)

def run_subtask_1(llm, text: str, use_mock: bool = False) -> str:
    """Sub-Task 1: Structured Executive Summarization Chain"""
    if use_mock or llm is None:
        return (
            "Summary:\n"
            "- Revenue & Growth: Revenue declined 4% YoY to $4.2B driven by macro disruptions.\n"
            "- Debt & Balance Sheet: Net debt rose to $1.8B post-acquisition; Debt-to-EBITDA reached 3.8x.\n"
            "- Guidance & Outlook: Full-year EBITDA guidance reduced by 150 bps due to supply chain friction."
        )
    prompt = ChatPromptTemplate.from_template(
        "Provide a structured executive financial summary evaluating key domain metrics. Format strictly as:\n"
        "Summary:\n"
        "- Revenue & Growth: <Key Finding>\n"
        "- Debt & Balance Sheet: <Key Finding>\n"
        "- Guidance & Outlook: <Key Finding>\n\n"
        "Input Text:\n{input}"
    )
    chain = prompt | llm
    return chain.invoke({"input": text}).content

def run_subtask_2(llm, text: str, use_mock: bool = False) -> str:
    """Sub-Task 2: Credit Risk Metrics & Sentiment Triage Chain"""
    if use_mock or llm is None:
        return (
            "Risk Triage Level: High\n"
            "Market Sentiment: Bearish\n"
            "Domain Metrics Assessment:\n"
            "- Debt-to-EBITDA Ratio Concern: 3.8x ratio exceeds prudent target threshold of 3.0x.\n"
            "- Liquidity Status ($ Cash): Stable cash balance of $600M provides short-term buffer.\n"
            "- Credit Risk Indicators: Leverage expansion and guidance markdown increase default sensitivity.\n"
            "- Covenant Breach Warning: Yes - High concern regarding maximum leverage covenant ceilings."
        )
    prompt = ChatPromptTemplate.from_template(
        "Analyze market sentiment and evaluate specific credit risk indicators. Format strictly as:\n"
        "Risk Triage Level: <Low/Medium/High/Critical>\n"
        "Market Sentiment: <Bearish/Neutral/Bullish>\n"
        "Domain Metrics Assessment:\n"
        "- Debt-to-EBITDA Ratio Concern: <Assessment>\n"
        "- Liquidity Status ($ Cash): <Assessment>\n"
        "- Credit Risk Indicators: <Assessment>\n"
        "- Covenant Breach Warning: <Yes/No - Assessment>\n\n"
        "Input Text:\n{input}"
    )
    chain = prompt | llm
    return chain.invoke({"input": text}).content

def run_subtask_3(llm, text: str, use_mock: bool = False) -> str:
    """Sub-Task 3: Covenant Actions & Audit Chain"""
    if use_mock or llm is None:
        return (
            "Recommended Actions:\n"
            "1. Action 1 (Covenant Compliance Audit): Issue formal compliance certificate request to verify 3.8x ratio against master credit facility limits.\n"
            "2. Action 2 (Liquidity & Leverage Mitigation): Require borrower to present 12-month debt de-leveraging plan and capital expenditure restrictions.\n"
            "3. Action 3 (Watchlist & Underwriting Review): Move corporate borrower to Credit Watchlist Grade B and increase monitoring frequency to monthly."
        )
    prompt = ChatPromptTemplate.from_template(
        "Extract the top 3 immediate credit risk audit and covenant compliance actions. Format strictly as:\n"
        "Recommended Actions:\n"
        "1. Action 1 (Covenant Compliance Audit)\n"
        "2. Action 2 (Liquidity & Leverage Mitigation)\n"
        "3. Action 3 (Watchlist & Underwriting Review)\n\n"
        "Input Text:\n{input}"
    )
    chain = prompt | llm
    return chain.invoke({"input": text}).content

if st.button("Execute All 3 Sub-Tasks Concurrently", type="primary"):
    is_valid, validation_msg = validate_inputs(api_key_input, input_payload, use_mock=use_mock_mode)
    
    if not is_valid:
        logger.warning(f"Validation failed: {validation_msg}")
        st.error(f"Validation Error: {validation_msg}")
    else:
        try:
            status_text = "Executing mock multi-task pipeline..." if use_mock_mode else "Executing multi-task NLP pipeline concurrently via ThreadPoolExecutor..."
            with st.spinner(status_text):
                llm = None
                if not use_mock_mode:
                    logger.info("Initializing ChatOpenAI instance with timeout boundaries...")
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        api_key=api_key_input.strip(),
                        temperature=0.1,
                        timeout=30.0,
                        max_retries=2
                    )
                
                cleaned_text = input_payload.strip()

                # Concurrent Multi-Threaded Execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future1 = executor.submit(run_subtask_1, llm, cleaned_text, use_mock_mode)
                    future2 = executor.submit(run_subtask_2, llm, cleaned_text, use_mock_mode)
                    future3 = executor.submit(run_subtask_3, llm, cleaned_text, use_mock_mode)

                    res1 = future1.result()
                    res2 = future2.result()
                    res3 = future3.result()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("1. Executive Summary")
                    st.success(res1)
                with col2:
                    st.subheader("2. Credit Risk Metrics")
                    st.warning(res2)
                with col3:
                    st.subheader("3. Covenant Actions")
                    st.info(res3)

                logger.info("All 3 sub-tasks executed concurrently successfully.")
        except Exception as global_err:
            logger.critical(f"Global pipeline failure: {str(global_err)}")
            st.error(f"Global Execution Error: {str(global_err)}")

st.markdown("---")
st.caption("Powered by OpenAI GPT-4o-mini via LangChain")
