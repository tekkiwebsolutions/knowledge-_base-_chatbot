# app.py

import streamlit as st
from data.main import rag_chain

# Function for generating LLM response
# def generate_response(input):
#     result = rag_chain.invoke(input)
#     return result
def generate_response(input):
    # Preprocess the input question to include relevant context
    input_with_context = f"Context: \nQuestion: {input}"

    result = rag_chain.invoke(input_with_context)
    response = result.split("Answer: ")[1].strip()  # Extract the generated answer

    return response
# Set page configuration
st.set_page_config(page_title="TWS HeroBot")

# Display chat interface
with st.sidebar:
    st.title('Tekki Websolutions HeroBot')

# Store chat messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, How Can i Help you"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.text_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": input})
    with st.spinner("Getting your answer from the TWS..."):
        response = generate_response(input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

