from flask import Flask, render_template, request, jsonify, session
import warnings
warnings.filterwarnings("ignore")
import os
import uuid
import markdown
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

app = Flask(__name__)

# --- CONFIGURATION FROM ENVIRONMENT VARIABLES ---
# CRITICAL: Flask Secret Key for session management. MUST be set in production!
app.secret_key = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    # In development, you might use a default, but for production, this should always be set.
    # For local development, you could set this in a .env file (see python-dotenv).
    print("WARNING: FLASK_SECRET_KEY environment variable not set. Using a default for development.")
    app.secret_key = 'your-very-strong-default-secret-key-for-dev-ONLY'
    # In a real production scenario, you might want to exit or raise an error here
    # to prevent running with an insecure default.
    # raise ValueError("FLASK_SECRET_KEY environment variable is not set. It is required for production.")

# Paths for vector store and notes directory
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vectorstore_gemini")
NOTES_PATH = os.getenv("NOTES_PATH", "notes")

# LLM Model name
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash-latest")


# Global variables to store the RAG chain
rag_chain = None
is_initialized = False

def setup_api_key():
    """Sets up the Google API key from environment variables."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set. Chatbot initialization may fail.")
        return False
    os.environ["GOOGLE_API_KEY"] = google_api_key # Set it for langchain to pick up
    return True

def process_markdown(text):
    """Converts markdown text to HTML."""
    return markdown.markdown(text)

def create_or_load_vector_store():
    """Creates or loads the vector store from documents."""
    global is_initialized

    try:
        # Check if vector store exists
        if os.path.exists(VECTOR_STORE_PATH):
            print(f"Loading vector store from {VECTOR_STORE_PATH}...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded.")
            return vector_store
        else:
            print("Vector store not found. Creating a new one...")
            if not os.path.exists(NOTES_PATH) or not os.listdir(NOTES_PATH):
                print(f"No notes found in {NOTES_PATH}. Creating a dummy document.")
                # Create a dummy document if no notes are found
                # This ensures the vector store creation doesn't fail if 'notes' is empty
                dummy_content = "This is a dummy document for chatbot initialization. Please add your actual knowledge base files to the 'notes' directory."
                documents = [Document(page_content=dummy_content, metadata={"source": "dummy_document"})]
            else:
                print(f"Loading documents from {NOTES_PATH}...")
                loader = DirectoryLoader(NOTES_PATH, glob="**/*.txt", show_progress=True)
                documents = loader.load()
                print(f"Loaded {len(documents)} documents.")

            if not documents:
                print("No documents loaded from notes directory. Cannot create vector store.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} chunks.")

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local(VECTOR_STORE_PATH)
            print("Vector store created and saved.")
            return vector_store
    except Exception as e:
        print(f"Error creating or loading vector store: {e}")
        is_initialized = False
        return None

def initialize_chatbot():
    """Initializes the RAG chatbot components."""
    global rag_chain, is_initialized

    if not setup_api_key():
        is_initialized = False
        return

    vector_store = create_or_load_vector_store()
    if vector_store is None:
        is_initialized = False
        return

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the user's question based on the provided context.
    If you cannot find the answer in the context, politely state that you don't have enough information.

    Context:
    {context}

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)
    is_initialized = True
    print("Chatbot initialized successfully!")

# Initialize chatbot on startup
# This will run once when the Flask app starts
with app.app_context():
    initialize_chatbot()

@app.route('/')
def index():
    """Renders the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages and returns responses from the RAG chain."""
    if not is_initialized:
        return jsonify({'error': 'Chatbot is not initialized. Please check your API key and notes directory.'}), 500

    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        show_sources = data.get('show_sources', False)

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Get response from RAG chain
        response = rag_chain.invoke({"input": user_message})

        # Process the response text to convert markdown to HTML
        processed_response = process_markdown(response["answer"])

        result = {
            'response': processed_response
        }

        # Only include sources if requested
        if show_sources:
            result['sources'] = [
                {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'source': doc.metadata.get('source', 'N/A')
                }
                for doc in response.get("context", [])
            ]

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check if the chatbot is ready."""
    return jsonify({'initialized': is_initialized})

if __name__ == '__main__':
    app.run(debug=True) # For local development, set debug=False for production testing
