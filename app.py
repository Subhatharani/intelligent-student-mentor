import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from html_templates import css, bot_template, user_template

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks using OpenAI embeddings."""
    if not text_chunks:
        return None
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain."""
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_with_llm(prompt_template_str, text_input):
    """Generates content using a specific prompt and the LLM from the conversation chain."""
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        return "Please process a document first."
    
    llm = st.session_state.conversation.combine_docs_chain.llm_chain.llm
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.run(text_input)
    return response

# --- Main Application ---

def main():
    """Main function to run the Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="AI Book Tutor", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "insights" not in st.session_state:
        st.session_state.insights = ""
    if "quiz" not in st.session_state:
        st.session_state.quiz = ""

    st.header("AI Book Summary + Explanation Tutor :books:")

    with st.sidebar:
        st.subheader("Your Book")
        pdf_docs = st.file_uploader(
            "Upload your book PDF here and click 'Process'", 
            accept_multiple_files=False, 
            type="pdf"
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your book... this might take a moment"):
                    # Reset states
                    st.session_state.raw_text = get_pdf_text([pdf_docs])
                    text_chunks = get_text_chunks(st.session_state.raw_text)
                    
                    if text_chunks:
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.summary = ""
                        st.session_state.insights = ""
                        st.session_state.quiz = ""
                        st.success("Processing complete! You can now interact with the book.")
                    else:
                        st.error("Could not extract text from the PDF. Please try another file.")
            else:
                st.warning("Please upload a PDF file first.")

    # Main content tabs
    if st.session_state.conversation:
        tab1, tab2, tab3, tab4 = st.tabs(["Chat with Book", "Summary", "Chapter Insights", "Quiz Generator"])

        # --- Chat Tab ---
        with tab1:
            st.subheader("Chat with the Book")
            user_question = st.text_input("Ask any question about the book's content:")
            if user_question:
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history = response['chat_history']
                
                # Display chat history
                for i, message in enumerate(st.session_state.chat_history):
                    template = user_template if i % 2 == 0 else bot_template
                    st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        # --- Summary Tab ---
        with tab2:
            st.subheader("Book Summary")
            if st.button("Generate Full Summary"):
                with st.spinner("Generating summary..."):
                    prompt = "Based on the entire text of the book provided, create a comprehensive summary. Cover the main plot, key themes, major characters, and the overall message or conclusion. The summary should be detailed enough to give a full understanding of the book. Here is the book content:\n\n{text}"
                    st.session_state.summary = generate_with_llm(prompt, st.session_state.raw_text)
            if st.session_state.summary:
                st.markdown(st.session_state.summary)
        
        # --- Insights Tab ---
        with tab3:
            st.subheader("Chapter-wise Insights")
            if st.button("Generate Chapter Insights"):
                with st.spinner("Generating insights..."):
                    prompt = "Analyze the provided book text and generate chapter-wise insights. For each major section or chapter, identify the key events, character developments, and thematic elements. Present it in a structured format. Here is the book content:\n\n{text}"
                    st.session_state.insights = generate_with_llm(prompt, st.session_state.raw_text)
            if st.session_state.insights:
                st.markdown(st.session_state.insights)

        # --- Quiz Tab ---
        with tab4:
            st.subheader("Auto Quiz Generator")
            if st.button("Generate Quiz"):
                with st.spinner("Generating quiz..."):
                    prompt = "Create a quiz based on the provided book text. It should include 5 multiple-choice questions (with 4 options each, clearly marking the correct answer) and 3 short-answer questions. The questions should cover a range of topics from the book. Here is the book content:\n\n{text}"
                    st.session_state.quiz = generate_with_llm(prompt, st.session_state.raw_text)
            if st.session_state.quiz:
                st.markdown(st.session_state.quiz)
    else:
        st.info("Please upload and process a book to enable the features.")


if __name__ == '__main__':
    main()
