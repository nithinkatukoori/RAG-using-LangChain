import streamlit as st
from retrive import answer
from retrive import to_markdown


st.title("RAG System on “Leave No Context Behind” Paper")

#Taking prompt from user interface
prompt=st.text_area("Type your Question Here")



#if button clicked in UI
if st.button('Submit')==True:
    res=answer(prompt)
    
    st.write(res)