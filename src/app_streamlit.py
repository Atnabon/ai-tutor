import os
import streamlit as st
from pathlib import Path
import tempfile
from qa_pipeline import create_vectorstore, build_rag_chain, build_summary_chain

# Set up the Streamlit page
st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("📘 AI Tutor - Learn From Your Own Books")

# Sidebar: Instructions and Uploaded Files List
st.sidebar.header("Instructions")
st.sidebar.markdown("""
**Please name your PDF files using the format:**  
`Grade <number> <Subject>`  
**Example:** `Grade 9 Biology.pdf`  
This helps the AI organize and reference your materials accurately.
""")

# Sidebar: File uploader for PDFs
uploaded_files = st.sidebar.file_uploader(
    "Upload a new textbook/note (PDF)", type="pdf", accept_multiple_files=True, key="sidebar_upload"
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded!")
    # Save files temporarily
    temp_dir = Path(tempfile.mkdtemp())
    saved_files = []
    for f in uploaded_files:
        filepath = temp_dir / f.name
        with open(filepath, "wb") as out_file:
            out_file.write(f.read())
        saved_files.append(filepath)

    # Sidebar: Display list of uploaded files
    st.sidebar.subheader("Uploaded Files")
    for f in saved_files:
        st.sidebar.write(f.name)

    # Build retriever and RAG chain
    with st.spinner("Indexing..."):
        retriever = create_vectorstore(saved_files)
    rag_chain = build_rag_chain(retriever)
        

    # Main page: Ask questions or request summarization
    st.subheader("Ask your textbook a question:")
    user_q = st.text_input("Enter your question")

    if st.button("Get Answer") and user_q:
        with st.spinner("Thinking..."):
            # Get answer from the RAG chain
            response = rag_chain.invoke(user_q)
            st.markdown("### ✨ Answer")
            st.write(response.content)

    st.subheader("Or summarize a topic from your textbook:")
    user_s = st.text_input("Enter the topic to summarize")
    summarize_file = st.selectbox(
        "Select a file to summarize", [f.name for f in saved_files]
    )
    if st.button("Summarize") and summarize_file:
        with st.spinner("Summarizing..."):
            # Load and chunk the selected file
            file_path = [f for f in saved_files if f.name == summarize_file][0]
            summary_chain = build_summary_chain(file_path)
            summary = summary_chain.invoke(user_s)
            
            st.markdown("### 📚 Summary")
            st.write(summary.content)

    





