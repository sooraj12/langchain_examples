# Basic example: prompt + model + output parser
# pip install –upgrade –quiet langchain-core langchain-community langchain-openai

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_openai.llms import OpenAI

# Prompt
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
# prompt_value = prompt.invoke({"topic": "ice cream"})

# Model. In this case our model is a ChatModel, meaning it will output a BaseMessage
model = ChatOpenAI(model="gpt-4")
# message = model.invoke(prompt_value)
# If our model was an LLM, it would output a string.
# llm = OpenAI(model="gpt-3.5-turbo-instruct")
# llm.invoke(prompt_value)

# Output parser
output_parser = StrOutputParser()
# output_parser.invoke(message)

# Entire Pipeline
# The prompt component takes the user input, which is then used to construct a PromptValue 
# after using the topic to construct the prompt.
# The model component takes the generated prompt, and passes into the OpenAI LLM model for evaluation. 
# The generated output from the model is a ChatMessage object.
# Finally, the output_parser component takes in a ChatMessage, and transforms this into a Python string, 
# which is returned from the invoke method.
chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})

# invoke
# stream
# batch
# async

# LLM instead of chat model
# If we want to use a completion endpoint instead of a chat endpoint
# llm = OpenAI(model="gpt-3.5-turbo-instruct")

# Runtime configurability
# If we wanted to make the choice of chat model or LLM configurable at runtime:
# from langchain_core.runnables import ConfigurableField
# configurable_model = model.configurable_alternatives(
#     ConfigurableField(id="model"), 
#     default_key="chat_openai", 
#     openai=llm,
#     anthropic=anthropic,
# )
# configurable_chain = (
#     {"topic": RunnablePassthrough()} 
#     | prompt 
#     | configurable_model 
#     | output_parser
# )
# configurable_chain.invoke(
#     "ice cream", 
#     config={"model": "openai"}
# )
# stream = configurable_chain.stream(
#     "ice cream", 
#     config={"model": "anthropic"}
# )
# for chunk in stream:
#     print(chunk, end="", flush=True)
# configurable_chain.batch(["ice cream", "spaghetti", "dumplings"])

# Logging

# Fallbacks
# If we wanted to add fallback logic, in case one model API is down:
# fallback_chain = chain.with_fallbacks([anthropic_chain])
# fallback_chain.invoke("ice cream")
# await fallback_chain.ainvoke("ice cream")
# fallback_chain.batch(["ice cream", "spaghetti", "dumplings"])