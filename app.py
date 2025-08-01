# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_system import retrieve_relevant_context # Import the RAG function

# Load the LLM and tokenizer
# Using a freely available model that doesn't require authentication
@st.cache_resource
def load_model():
    print("Loading LLM and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(query):
    """
    Generates a safe and informative response using RAG and the LLM.
    """
    # Add a critical safety disclaimer
    disclaimer = (
        "**Disclaimer:** This information is for educational purposes only "
        "and should not be considered medical advice. Always consult a "
        "healthcare professional for diagnosis and treatment."
    )

    # Get relevant context from our RAG system
    context = retrieve_relevant_context(query)
    
    # If we have context, use it directly instead of relying on LLM generation
    if context and len(context.strip()) > 50:
        # Format the context nicely
        context_parts = context.split('\n\n')
        relevant_info = context_parts[0] if context_parts else context
        
        # Create a more direct response using the retrieved information
        response_text = f"Here's what I found about '{query}':\n\n{relevant_info}\n\nThis information is from our medical knowledge base. For specific medical advice, please consult a healthcare professional."
    else:
        # Fallback if no relevant context found
        response_text = f"I couldn't find specific information about '{query}' in our medical database. Please consult a healthcare professional for accurate medical information."
    
    return f"{disclaimer}\n\n{response_text}"

# Continuing in app.py from the previous step
st.title("Medical Assistant Chatbot")
st.write("I can answer general questions about medical conditions. "
         "**Always consult a professional for medical advice.**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = generate_response(prompt)
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response}) 