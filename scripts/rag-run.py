import streamlit as st
import feynman_llm_rag as flr

# Title of the app
st.title("Feynman Query App")

# Text input for the user query
query = st.text_input("Enter your query:", "Speak as Feynman for fun.")

# When the user submits the query, process it
if st.button("Get Answer"):
    # Get the answer from the rag_query function
    answer = flr.rag_query(query)
    
    # Display the query and the answer
    st.write(f"**Query:** {query}")
    st.write(f"**Answer:** {answer}")
