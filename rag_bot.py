
import os
from dotenv import load_dotenv #dotenv for api config
load_dotenv()

# configuring groq n google api key
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain_groq import ChatGroq #GROQ ISTHE BEST OMGOMGOMGOMGOMGOMGOMGOMG
from langchain_text_splitters import RecursiveCharacterTextSplitter # to create chunks
from langchain.chains.retrieval import create_retrieval_chain # retrieval chain for answers
from langchain.chains.combine_documents import create_stuff_documents_chain # document chain 
from langchain_core.prompts import ChatPromptTemplate # providing template to the model to formulate answers as you desire
from langchain_community.vectorstores import FAISS # to store vector embeddings that are created
from langchain_community.document_loaders import PyPDFDirectoryLoader # self explanatory
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Gen AI embedding. use embedding 004 
import streamlit as st

st.title("ENROLLMENT ASSISTANT CHATBOT")

llm_model = ChatGroq(groq_api_key= groq_api_key, 
                    model="Llama3-8b-8192") # models with more parameters (like the llama 3 70b) are better but 
                                            # a) RPM (request per minute) is low 
                                            # b) tokens per second can be low depending on availability

user_prompt=ChatPromptTemplate.from_template (
"""
Do not mention "based on/according to context/text etc". Maintain a friendly tone.
Answers regarding course details should have ONLY (if applicable) Course Category, Course code, Name, Credits and pre requisites.
<context>
{context}
<context>
Questions:{input}
"""
)   # use magic words like please and be polite to your model

def vector_embedding():

    if "vectors" not in st.session_state:
        
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
        st.session_state.loader=PyPDFDirectoryLoader("./clg_docs") ## get all docs )
        st.session_state.docs=st.session_state.loader.load() ## load all docs
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## create chunks
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector embeddings

with st.spinner("Embedding docs :)"):
    vector_embedding()
# st.write(st.session_state.docs) -> to see if all docs have loaded properly

# streamlit configuration 
st.chat_message("assistant").markdown("Hello there!! I am your virtual assistant for enrollments!! Ask me academic questions and ill try to answer!!")
st.chat_message("assistant").markdown("""I am still experimental and prone to mistakes. If you want to know the details of a course,
                                       **format your prompt** with\n\n <Subject Code/Name> course details\n\n for an accurate description. 
                                      Seeking other information does not involve any prompt constraints. Have a great Day!!""")

if 'history' not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
    

if user_input:= st.chat_input("Ask me Something!"):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.history.append({"role": "user", "message": user_input})

# try except is important here bc the model raises an error if the text input box is empty.
try:
    document_chain=create_stuff_documents_chain(llm_model,user_prompt) # model interacts with doc to retrieve answers
    retriever=st.session_state.vectors.as_retriever() # vectors stored in session state converted to retriever obj
    retrieval_chain=create_retrieval_chain(retriever,document_chain) # response retriever pipeline is setup
    response=retrieval_chain.invoke({'input':user_input}) # input is passed and the retriever pipeline is "invoked" (called)
                                                          # the pipeline allows the llm to interact with the document, fetch relevant responses
                                                          # and returns them in response["answer"]
                                                          # response["context"] contains the context of the answer.

    with st.chat_message("bot"):
        st.markdown(response['answer'])
    st.session_state.history.append({"role": "bot", "message": response['answer']})

except TypeError as e:
    answer = "Ask Away!!"


