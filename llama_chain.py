from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    "Explain in simple terms what {disease} is and how it affects tomato plants."
)

chain = prompt | llm | StrOutputParser()

def get_llama_response(disease):
    return chain.invoke({"disease": disease})