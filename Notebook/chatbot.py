import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import pinecone
import tempfile

load_dotenv('.env')

# Inizializza Pinecone
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)

# Imposta variabili di ambiente per LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "ls__0554eeed658141b4a601195abd6c737d"

class chatbt:
    def __init__(self):
        self.chat_history = []
        self.qa, self.vector_store = self.load_db("stuff", 4)

    def load_db(self, chain_type, k):
        embeddings = OpenAIEmbeddings()
        vector_store = Pinecone.from_existing_index('embedding-bandi', embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name='ft:gpt-3.5-turbo-1106:links:gpt-3-5-signorile:8S8SNEPI', temperature=0.5),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa, vector_store
    
    def load_pdf(self, uploaded_file):
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
        pdf_loader = PyPDFLoader(path)
        pdf_text = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(pdf_text)
        embeddings = OpenAIEmbeddings()
        index = pinecone.Index("embedding-bandi")
        embeddedvector = embeddings.embed_query(docs)
        index.upsert(vectors=[embeddedvector])
        
        chatbt_instance.qa, chatbt_instance.vector_store = chatbt_instance.load_db("stuff", 4)

        return st.success("Documento PDF caricato con successo!")

# Streamlit code
st.title('Chatbot Bandi in corso Regione Puglia')
chatbt_instance = chatbt()

uploaded_file = st.file_uploader("File upload", type="pdf")
if uploaded_file:
     chatbt_instance.load_pdf(uploaded_file)

# messaggio di benvenuto
message = st.chat_message("assistant")


user_input = st.chat_input("Inserisci la tua domanda:")

if user_input:
    message_input = st.chat_message("user")
    message_input.write(user_input)
    result = chatbt_instance.qa({"question": user_input, "chat_history": chatbt_instance.chat_history})
    chatbt_instance.chat_history.append([(user_input, result["answer"])])
    chatbt_instance.qa, chatbt_instance.vector_store = chatbt_instance.load_db("stuff", 4)
    chatbt_instance.answer = result['answer'] 
    # Visualizza la risposta del chatbot all'interno della chat
    
    message.write(chatbt_instance.answer)

