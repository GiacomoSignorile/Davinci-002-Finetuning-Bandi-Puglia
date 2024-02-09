import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import Pinecone as PineconeStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import tempfile
import time
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv('.env')

# Inizializza Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Imposta variabili di ambiente per LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"

class chatbt:
    pdf_caricato = False

    def __init__(self):
        self.chat_history = []
        self.qa = self.load_db("stuff", 4)
        self.pdf_caricato = True
        self.vector_store = self.load_vector_store()

    def load_db(self, chain_type, k):
        vector_store = self.load_vector_store()
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name='ft:gpt-3.5-turbo-1106:links:gpt-3-5-signorile:8S8SNEPI', temperature=0.5),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa 

    def load_vector_store(self):
        embeddings = OpenAIEmbeddings()
        directory = 'Documenti/docs/chroma/'
        # Utilizza Chroma solo se il PDF non è stato caricato
        if self.pdf_caricato == True:
            vector_store = Chroma(persist_directory= directory,embedding_function = embeddings)
        else:
            indexname='embedding-bandi'
            vector_store = PineconeStore(index_name = indexname, embedding = embeddings)
        return vector_store
        
    
    def load_pdf(self, uploaded_file):
        directory = 'Documenti/docs/chroma/'
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
        pdf_loader = PyPDFLoader(path)
        pdf_text = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(pdf_text)
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory= directory)

        chatbt_instance.vector_store = vector_store
        chatbt_instance.qa = chatbt_instance.load_db("stuff", 4)
        self.pdf_caricato = True
        return st.success("Documento PDF caricato con successo!"),vector_store 

# Streamlit code
st.title('Chatbot Bandi in corso Regione Puglia')
chatbt_instance = chatbt()

uploaded_file = st.file_uploader("File upload", type="pdf")
if uploaded_file:
     chatbt_instance.load_pdf(uploaded_file)

# messaggio di benvenuto
with st.chat_message('assistant'):
     st.write("Ciao, sono il tuo assistente personale personalizzato per rispondere a domande relative ai bandi in corso della regione Puglia!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Inserisci la tua domanda:"):
    with st.chat_message("user"):
        st.markdown(user_input)

    result = chatbt_instance.qa({"question": user_input, "chat_history": chatbt_instance.chat_history})
    chatbt_instance.chat_history.append([(user_input, result["answer"])])
    chatbt_instance.qa = chatbt_instance.load_db("stuff", 4)
    chatbt_instance.vector_store = chatbt_instance.load_vector_store()
    chatbt_instance.answer = result['answer']
     
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = chatbt_instance.answer
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

