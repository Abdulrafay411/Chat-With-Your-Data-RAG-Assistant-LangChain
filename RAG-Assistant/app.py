import os
import streamlit as st
from rag_pipeline import load_and_process_pdf, create_vector_store, get_rag_chain
import tempfile
import litellm

st.set_page_config(page_title="Chat With Your Data", page_icon="📄")

st.title("📄 Chat with your PDF (RAG)")
st.write("Upload a document and ask questions about it!")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Provider API Key", type="password", help="Enter the API key for your chosen model provider (e.g., Anthropic, Google, Groq, etc.)")
    
    chat_models = ["Custom..."] + sorted(litellm.model_list)
    default_chat_idx = chat_models.index("anthropic/claude-3-haiku-20240307") if "anthropic/claude-3-haiku-20240307" in chat_models else 0
    selected_chat = st.selectbox("Select Chat Model", chat_models, index=default_chat_idx)
    if selected_chat == "Custom...":
        chat_model = st.text_input("Custom Chat Model Name", value="anthropic/claude-3-haiku-20240307", help="Format: provider/model")
    else:
        chat_model = selected_chat
        
    emb_models = ["Custom..."] + sorted(list(litellm.all_embedding_models))
    default_emb_idx = emb_models.index("text-embedding-3-small") if "text-embedding-3-small" in emb_models else 0
    selected_emb = st.selectbox("Select Embedding Model", emb_models, index=default_emb_idx)
    if selected_emb == "Custom...":
        embedding_model = st.text_input("Custom Embedding Model Name", value="text-embedding-3-small", help="Format: provider/model")
    else:
        embedding_model = selected_emb
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if not api_key:
    st.info("Please add your API key in the sidebar to continue.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

if uploaded_file and (st.session_state.rag_chain is None or st.session_state.last_file_name != uploaded_file.name):
    st.session_state.last_file_name = uploaded_file.name
    with st.spinner("Processing document... This might take a minute."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            splits = load_and_process_pdf(tmp_file_path)
            vectorstore = create_vector_store(splits, embedding_model, api_key)
            st.session_state.rag_chain = get_rag_chain(vectorstore, chat_model, api_key)
            st.success("Document processed! You can now ask questions.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            os.remove(tmp_file_path)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.rag_chain is None:
        st.error("Please upload a document first.")
    else:
        chat_history = []
        for msg in st.session_state.messages:
            role = "human" if msg["role"] == "user" else "assistant"
            chat_history.append((role, msg["content"]))

        st.chat_message("user").markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                answer = response["answer"]
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})
