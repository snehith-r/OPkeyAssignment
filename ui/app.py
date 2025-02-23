import streamlit as st
import requests

# Define the FastAPI backend endpoint
API_URL = "http://fastapi:8000/ask_question/"

# Streamlit UI
# Streamlit UI
def main():
    st.title("🐶 Intelligent Dog Breed Assistant")
    st.write("Ask any question about dog breeds, and I'll fetch the best answer for you!")
    
    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # User input
    query = st.text_input("Enter your question:", "")
    
    if st.button("Ask"):  
        if query:
            response = requests.post(API_URL, json={"query": query, "search_strategy": "keyword"})
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No response received.")
                st.session_state.history.append({"question": query, "answer": answer})
                
                # Display the latest answer immediately
                st.write("### Response")
                st.write(f"**Q:** {query}")
                st.write(f"**A:** {answer}")
                st.write("---")
            else:
                st.error("Failed to fetch response from the backend.")
    
    # Display conversation history
    st.write("### Conversation History")
    for chat in reversed(st.session_state.history):
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        st.write("---")

if __name__ == "__main__":
    main()