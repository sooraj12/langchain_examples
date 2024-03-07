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


