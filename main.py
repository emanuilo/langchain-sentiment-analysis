import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a JSON object."""

    def parse(self, text: str) -> dict:
        result = {"result": text}
        return json.dumps(result)


template = """You are a model for sentiment analysis. You are given a sentence and you have to
predict whether it is positive or negative. Only return the sentiment in format POSITIVE or
NEGATIVE and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([("system", template), ("human", human_template)])
chain = chat_prompt | ChatOpenAI() | JsonOutputParser()

response = chain.invoke({"text": "This product is good!"})
print(response)
