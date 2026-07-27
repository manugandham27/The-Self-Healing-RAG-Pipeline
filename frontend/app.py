"""Streamlit Interactive Visual Dashboard for Self-Healing RAG Pipeline."""

import sys
import os
import time
import json
import streamlit as st

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from selfhealing_rag.config import settings
from selfhealing_rag.llm_client import AnthropicLLMClient, MockLLMClient
from selfhealing_rag.vector_store import ChromaVectorStore
from selfhealing_rag.retriever import Retriever
from selfhealing_rag.generator import Generator
from selfhealing_rag.critic import Critic
from selfhealing_rag.reformulator import QueryReformulator
from selfhealing_rag.orchestrator import Orchestrator

st.set_page_config(
    page_title="Self-Healing RAG Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .attempt-card-accepted {
        border-left: 5px solid #10B981;
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .attempt-card-rejected {
        border-left: 5px solid #EF4444;
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .attempt-card-fallback {
        border-left: 5px solid #F59E0B;
        background-color: #FFFBEB;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .badge-accepted {
        background-color: #10B981;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-rejected {
        background-color: #EF4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-fallback {
        background-color: #F59E0B;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_orchestrator():
    """Initialize and cache pipeline dependencies."""
    if settings.anthropic_api_key:
        llm_client = AnthropicLLMClient()
    else:
        llm_client = MockLLMClient()

    vector_store = ChromaVectorStore()
    retriever = Retriever(vector_store=vector_store)
    generator = Generator(llm_client=llm_client)
    critic = Critic(llm_client=llm_client)
    reformulator = QueryReformulator(llm_client=llm_client)

    return Orchestrator(
        retriever=retriever,
        generator=generator,
        critic=critic,
        reformulator=reformulator
    )


# Title and Header
st.markdown('<div class="main-title">🛡️ Self-Healing RAG Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation that critiques its own answers, detects hallucinations, and reformulates queries autonomously.</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    max_retries = st.slider("Max Retries", min_value=1, max_value=5, value=2, help="Maximum number of query reformulation retries")
    enable_self_healing = st.toggle("Enable Self-Healing Loop", value=True, help="Toggle between Self-Healing RAG and Baseline single-pass RAG")
    
    st.divider()
    st.header("💡 Preset Queries")
    
    preset_1 = "What encryption and key management standards are required for vector databases?"
    preset_2 = "What are the exact container UID and root filesystem rules for Kubernetes deployments?"
    preset_3 = "What is the mandatory SOC 2 Type II audit frequency for enterprise AI models?" # Out of corpus / trick question
    
    if st.button("Sample Query 1 (Encryption)", use_container_width=True):
        st.session_state["query_input"] = preset_1
    if st.button("Sample Query 2 (Kubernetes Security)", use_container_width=True):
        st.session_state["query_input"] = preset_2
    if st.button("Sample Query 3 (Trick / Out-of-Scope)", use_container_width=True):
        st.session_state["query_input"] = preset_3

    st.divider()
    if st.button("📥 Ingest Sample Docs into ChromaDB", use_container_width=True):
        with st.spinner("Ingesting sample documents..."):
            try:
                from scripts.ingest import ingest_sample_docs
                ingest_sample_docs()
                st.success("Sample docs ingested successfully!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# Main Query Section
default_query = st.session_state.get("query_input", preset_1)
user_query = st.text_input("Enter your question:", value=default_query, key="user_query_key")

if st.button("🚀 Run Self-Healing Pipeline", type="primary", use_container_width=True):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        orchestrator = get_cached_orchestrator()
        orchestrator.max_retries = max_retries

        with st.spinner("Running retrieval, generation, and critique loop..."):
            start_t = time.time()
            response = orchestrator.run(query=user_query, enable_self_healing=enable_self_healing)
            elapsed = time.time() - start_t

        st.markdown("---")
        
        # Status Banner
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Status", response.status)
        with col_s2:
            st.metric("Total Attempts", response.total_attempts)
        with col_s3:
            st.metric("Execution Time", f"{elapsed:.2f}s")
        with col_s4:
            st.metric("Self-Healing Mode", "ON" if enable_self_healing else "OFF (Baseline)")

        st.subheader("🎯 Final Answer")
        st.info(response.answer)

        st.subheader("🔄 Visual Self-Healing Execution Trace")
        st.caption("Expand attempt cards below to inspect retrieval, generation, critic verdicts, and query reformulation steps.")

        for trace in response.traces:
            status = trace.status
            attempt_num = trace.attempt_number
            card_class = "attempt-card-accepted" if status == "ACCEPTED" else ("attempt-card-rejected" if status == "REJECTED" else "attempt-card-fallback")
            badge_class = "badge-accepted" if status == "ACCEPTED" else ("badge-rejected" if status == "REJECTED" else "badge-fallback")

            with st.expander(f"Attempt #{attempt_num} — [{status}] — Query: '{trace.query}'", expanded=True):
                st.markdown(f'<span class="{badge_class}">{status}</span>', unsafe_allow_html=True)
                st.markdown(f"**Search Query Used:** `{trace.query}`")

                # Retrieved chunks tab
                tab1, tab2, tab3 = st.tabs(["📚 Retrieved Chunks", "📝 Candidate Answer", "⚖️ Critic Verdict"])

                with tab1:
                    if trace.retrieved_chunks:
                        for c_idx, chunk in enumerate(trace.retrieved_chunks, 1):
                            st.markdown(f"**Passage [{c_idx}]** — *Source*: `{chunk.metadata.get('source', 'Unknown')}` — *Score*: `{chunk.score}`")
                            st.code(chunk.content, language="markdown")
                    else:
                        st.write("No chunks retrieved.")

                with tab2:
                    if trace.generation:
                        st.markdown(trace.generation.answer)
                        if trace.generation.citations:
                            st.markdown("**Citations Extracted:**")
                            for cit in trace.generation.citations:
                                st.write(f"- `[{cit.citation_index}]` Source: **{cit.source}** (Chunk ID: `{cit.chunk_id}`)")
                    else:
                        st.write("No generation output.")

                with tab3:
                    if trace.verdict:
                        v = trace.verdict
                        v_col1, v_col2 = st.columns(2)
                        with v_col1:
                            st.metric("Grounded Status", "✅ YES" if v.grounded else "❌ NO")
                        with v_col2:
                            st.metric("Critic Confidence", f"{v.confidence * 100:.0f}%")
                        
                        st.markdown(**Critic Reason:** {v.reason})
                        if v.unsupported_claims:
                            st.error("**Unsupported Claims Detected:**")
                            for uc in v.unsupported_claims:
                                st.write(f"- {uc}")
                    else:
                        st.write("No critic evaluation.")

                if trace.reformulation:
                    st.warning("⚡ **Query Reformulated for Next Attempt:**")
                    st.write(f"- **New Query:** `{trace.reformulation.reformulated_query}`")
                    st.write(f"- **Reformulation Rationale:** {trace.reformulation.reasoning}")

        # JSON Trace Inspector
        with st.expander("🔍 Developer Raw JSON Trace", expanded=False):
            st.json(response.model_dump())
