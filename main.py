import streamlit as st
import google.generativeai as gen_ai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure Streamlit page settings
st.set_page_config(
    page_title="BuddyBOT", #add page title
    page_icon=":brain:",  # add emoji
    layout="wide",  # Page layout option
)

# Display the chatbot's title on the page
st.title("🤖 ChatBot")

with st.chat_message("assistant"):# Adding a chat message block for the assistant input.
    st.markdown("Hello, I am your assistant. How can I help you?") # Displaying the assistent first wellcom line.

# Add the button to sidebar
if st.sidebar.button("New Chat"):
    # Clear the chat session
    st.session_state.chat_session = []
    st.session_state.new_chat_clicked = True
    st.empty()  # Clear the UI to remove chat messages

theme = st.sidebar.radio("Select Theme", ("Light", "Dark"))
st.sidebar.subheader("Chat History")
# Function to display chat history in the sidebar
def display_chat_history():
    for chat in st.session_state.chat_session:
        if chat["role"] == "user":
            st.sidebar.text(f"User: {chat['context']}")
        elif chat["role"] == "assistant":
            st.sidebar.text(f"Assistant: {chat['context']}")

# Set up Google Gemini-Pro AI model
GOOGLE_API_KEY=gen_ai.configure(api_key="AIzaSyCiPt8B5VpJnwb9ChD6abJ67hjnCu6gvCI")
model = gen_ai.GenerativeModel('gemini-pro')


# Function to load CSS styles
def load_css(file_name):
    with open(file_name, 'r') as f:
        css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load CSS styles
load_css('style.css')



# Apply theme based on selection
if theme == "Light":
    st.markdown(f'<body class="light">', unsafe_allow_html=True)
else:
    st.markdown(f'<body class="dark">', unsafe_allow_html=True)


# Function to translate roles from Gemini-Pro to Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
chat=model.start_chat(history=[])
def get_gemini_response(question):
    response=chat.send_message(question)
    return response.text
# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = []
    st.session_state.new_chat_clicked = False
# if the chat history is vide creat a new chat

# Display the chat history
for message in st.session_state.chat_session:
    if not st.session_state.new_chat_clicked:
        with st.chat_message(translate_role_for_streamlit(message["role"])):
            st.markdown(message["context"])

#extract the text from the pdf files
def get_pdf_text(pdf_docs):
    text = "" 
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file {pdf}: {e}")
    return text

#splitt the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#store he text chunks into vector data base using faiss
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

pdf_docs=['Banque_FR.pdf','banque_AR.pdf']#la base de connaissance
raw_text = get_pdf_text(pdf_docs)#extrairle text
text_chunks = get_text_chunks(raw_text)#splitt the text
get_vector_store(text_chunks)#store the information into faiss database

#creat a prompt for the ai model
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:"""
    model = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#generating a response after the user input the question
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings)#load the db
    docs = new_db.similarity_search(user_question)#search in the db for similarity
    # Use conversational chain to answer based on found documents
     # Use conversational chain to answer based on found documents
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        if response is not None and response["output_text"] != "answer is not available in the context":
            return response["output_text"]
        else:
            return False
    except Exception as e:
        print(f"An exception occurred: {e}")
        return False

# Input field for user's message
user_prompt = st.chat_input("Type your message here...")
if user_prompt:
    # Add user's message to chat session
    st.session_state.chat_session.append({"role": "user", "context": user_prompt})
    # Add user's message to chat history and display it
    st.chat_message("user").markdown(user_prompt)
    # Send user's message to Gemini-Pro and get the response
    bot_response = user_input(user_prompt)
    # Display bot's response
    if bot_response is not False:
        st.session_state.chat_session.append({"role": "assistant", "context": bot_response})
        # Display bot's response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
          
    else:
        # Send user's message to Gemini-Pro and get the response
        gemini_response=get_gemini_response(user_prompt)
        st.session_state.chat_session.append({"role": "assistant", "context": gemini_response})
        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
                response=gemini_response
                st.markdown(response)
    pass

display_chat_history()