import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma,Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import pinecone
import tempfile
import time
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
        vector_store = Chroma.from_documents(docs, embeddings)

        chatbt_instance.vector_store = vector_store
        chatbt_instance.qa = chatbt_instance.load_db("stuff", 4)

        return st.success("Documento PDF caricato con successo!"),vector_store

# Streamlit code
st.title('Chatbot Bandi in corso Regione Puglia')
chatbt_instance = chatbt()

uploaded_file = st.file_uploader("File upload", type="pdf")
if uploaded_file:
     chatbt_instance.load_pdf(uploaded_file)

# messaggio di benvenuto
with st.chat_message('assistant'):
     st.write("Ciao, sono il tuo assistente personlale personalizzato per rispondere a domande relative ai bandi in corso della regione Puglia!")

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
    chatbt_instance.qa, chatbt_instance.vector_store = chatbt_instance.load_db("stuff", 4)
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

