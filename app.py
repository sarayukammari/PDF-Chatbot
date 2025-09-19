import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7, max_output_tokens=1024)

def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=400)
    chunks = text_splitter.split_text(text)
    return chunks

def embeddings_text(chunks):
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def conversation_chain():
    prompt_template="""
    Answer the question as detailed as possible using the provided context, make sure to provide all details, if answer in not there in the provided context just say, "answer is not available in the provided context", dont provide any other information.
    Context: {context}
    Question: {question} """
    model=ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3, max_output_tokens=1024)
    prompt= PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_query(query):
    embedding= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore= FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    docs= vectorstore.similarity_search(query, k=4)
    chain= conversation_chain()
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return response
     

def main():
    st.header("PDF Chatbot with Google Generative AI")
    user_question= st.text_input("Enter your question")
    st.button("Ask")
    if user_question:
        with st.spinner("Generating response..."):
            response= user_query(user_question)
            st.text_area("Response", value=response["output_text"])
            
   
    with st.sidebar:
        st.subheader("Import PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if st.button("Process"):
           with st.spinner("Processing PDF..."):
               processed_pdf=process_pdf(uploaded_file)
               chunks=text_chunks(processed_pdf)
               embeddings_text(chunks)
               st.success("PDF processed successfully!")
               

if __name__ == "__main__":
     main()
    

