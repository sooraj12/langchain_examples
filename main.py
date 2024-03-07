from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_openai import ChatOpenAI
# from langchain import hub
# from langchain.agents import create_openai_functions_agent
# from langchain.agents import AgentExecutor

# initialize llm
llm = Ollama(model="llama2")
# llm.invoke("how can langsmith help with testing?")

# load the document
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# store embeddings in vector db
embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# setup retriever
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# setup search tool
search = TavilySearchResults()

# setup output parser
output_parser = StrOutputParser()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
# ])

tools = [retriever_tool, search]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

# prompt = hub.pull("hwchase17/openai-functions-agent")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# agent = create_openai_functions_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "how can langsmith help with testing?"})
# agent_executor.invoke({"input": "what is the weather in SF?"})
# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# agent_executor.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# document_chain.invoke({
#     "input": "how can langsmith help with testing?",
#     "context": [Document(page_content="langsmith can let you visualize test results")]
# })

# chain = prompt | llm | output_parser

# chain.invoke({"input": "how can langsmith help with testing?"})

# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})