import streamlit as st
from langchain_aws import ChatBedrock
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain.chains import RetrievalQA
import tempfile

# UI Title
st.title("📄 Document Q&A with Claude 3 (Bedrock)")

# File uploader
uploaded_file = st.file_uploader("Upload a text document", type=["txt"])

# Process uploaded document
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load and split document
    docs = TextLoader(tmp_path).load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create vector store
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create Claude 3 Sonnet model
    chat = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1")

    # QA Chain
    qa = RetrievalQA.from_chain_type(llm=chat, retriever=vectorstore.as_retriever())

    # Ask a question
    user_query = st.text_input("Ask a question about the document:")

    if user_query:
        with st.spinner("Searching and thinking..."):
            answer = qa.run(user_query)
        st.success("Answer:")
        st.write(answer)
