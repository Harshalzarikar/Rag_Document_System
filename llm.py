import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ["GOOGLE_API_KEY"]="AIzaSyAFx5DAUyOHT7jBfj4qopOtGKFlUUID0Vo"

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "You are a sarcastic senior engineer. Answer briefly."),
         ("user", "{input}")
    ]
)

parser=StrOutputParser()

chain= prompt| model|parser
response= chain.invoke({"input": "tell me  chetashree jagtap who is she"})
print(response)