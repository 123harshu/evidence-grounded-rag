import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Evidence-Grounded RAG", layout="wide")
st.title("🛡️ Evidence-Grounded AI Research Assistant")

tabs = st.tabs(["🔍 Search & Query", "📁 Document Management", "📊 Evaluation Dashboard"])

with tabs[0]:
    st.subheader("Query the Knowledge Base")
    query = st.text_input("Enter your technical research question:")
    top_k = st.slider("Top-K Passages to Retrieve", 1, 10, 4)
    
    if st.button("Ask Assistant", type="primary"):
        if query:
            res = requests.post(f"{API_URL}/query", json={"question": query, "top_k": top_k}).json()
            
            st.markdown("### Answer")
            if res["answer"] == "Insufficient evidence.":
                st.warning("⚠️ Insufficient evidence in the indexed documents to support an answer.")
            else:
                st.success(res["answer"])
            
            st.caption(f"⏱️ **Latency:** {res['latency_sec']}s")
            
            st.markdown("### Source Citations")
            for cit in res.get("citations", []):
                st.info(f"📄 {cit}")

            with st.expander("🔬 Inspect Retrieved Evidence Passages & RRF Scores"):
                for idx, p in enumerate(res.get("passages", [])):
                    st.write(f"**Passage {idx+1}** | Ref: `{p['full_ref']}` | Score: `{p['rrf_score']}`")
                    st.code(p["content"])

with tabs[1]:
    st.subheader("Ingest Research Documents")
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt", "md"])
    if uploaded_file and st.button("Ingest Document"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        res = requests.post(f"{API_URL}/ingest", files=files).json()
        st.success(f"Ingested '{res['filename']}' into {res['chunks_created']} vector chunks.")

with tabs[2]:
    st.subheader("System Evaluation Harness")
    if st.button("Run Offline Evaluation Suite"):
        with st.spinner("Running evaluation suite..."):
            metrics = requests.get(f"{API_URL}/evaluate").json()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Latency", f"{metrics['avg_latency_sec']}s")
            col2.metric("Retrieval Hit Rate", f"{metrics['hit_rate'] * 100}%")
            col3.metric("Refusal Accuracy", f"{metrics['refusal_accuracy'] * 100}%")
            
            st.markdown("### Detailed Test Logs")
            for item in metrics["detailed_results"]:
                color = "green" if item["status"] == "green" else ("orange" if item["status"] == "amber" else "red")
                st.markdown(f":{color}[**[{item['category'].upper()}]** Question: {item['question']}]")
                st.write(f"Response: *{item['answer']}* (Latency: {item['latency_sec']}s)")
                st.divider()
