import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import backoff
import os

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Obtén información instantánea de tus documentos

Este chatbot está construido utilizando el marco de Generación Aumentada por Recuperación (RAG), aprovechando el modelo ChatGPT de OpenAI. Procesa los documentos PDF cargados descomponiéndolos en fragmentos manejables, crea un almacén vectorial de búsqueda y genera respuestas precisas a las consultas del usuario. Este enfoque avanzado asegura respuestas de alta calidad y contextualmente relevantes para una experiencia de usuario eficiente y efectiva.

### Cómo Funciona

Sigue estos simples pasos para interactuar con el chatbot:

1. **Introduce tu clave API**: Necesitarás una clave API de OpenAI para que el chatbot pueda acceder a los modelos de OpenAI. Obtén tu clave API [aquí](https://beta.openai.com/signup/).

2. **Carga tus documentos**: El sistema acepta múltiples archivos PDF a la vez, analizando el contenido para proporcionar información completa.

3. **Haz una pregunta**: Después de procesar los documentos, haz cualquier pregunta relacionada con el contenido de tus documentos cargados para obtener una respuesta precisa.
""")

api_key = st.text_input("Introduce tu clave API de OpenAI:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Responde la pregunta tan detalladamente como sea posible a partir del contexto proporcionado. Asegúrate de proporcionar todos los detalles. Si la respuesta no está en
    el contexto proporcionado, simplemente di, "la respuesta no está disponible en el contexto", no proporciones una respuesta incorrecta\n\n
    Contexto:\n {context}?\n
    Pregunta: \n{question}\n

    Respuesta:
    """
    model = OpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=None)
def embed_query_with_retry(embedding_function, query):
    return embedding_function.embed_query(query)

def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Respuesta: ", response["output_text"])

def main():
    st.header("Chatbot clon de IA")

    user_question = st.text_input("Haz una pregunta sobre los archivos PDF", key="user_question")

    if user_question and api_key:  # Asegúrate de que se proporcionen la clave API y la pregunta del usuario
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menú:")
        pdf_docs = st.file_uploader("Carga tus archivos PDF y haz clic en el botón Enviar y Procesar", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Enviar y Procesar", key="process_button") and api_key:  # Verifica si se proporciona la clave API antes de procesar
            with st.spinner("Procesando..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Hecho")

if __name__ == "__main__":
    main()
