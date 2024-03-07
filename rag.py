# RAG Search Example
# pip install langchain docarray tiktoken
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# prompt template above takes in context and question as values to be substituted in the prompt. 
# Before building the prompt template, we want to retrieve relevant documents to the search and 
# include them as part of the context.
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings(),
)
# setup the retriever using an in memory store, which can retrieve documents based on a query.
retriever = vectorstore.as_retriever()
# This is a runnable component as well that can be chained together with other components, 
# but you can also try to run it separately:
# retriever.invoke("where did harrison work?")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | model | output_parser

chain.invoke("where did harrison work?")