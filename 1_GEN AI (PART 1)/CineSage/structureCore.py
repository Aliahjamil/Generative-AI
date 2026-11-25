from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=300
)

# schema for movie information
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str

# parser for movie information
parser = PydanticOutputParser(pydantic_object=Movie)

# Instantiation using from_messages
prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract movie information from the paragraph.
Use only facts provided in the paragraph.
Use null for missing optional values.
Return valid JSON matching this format:
{format_instructions}
"""),
    ("human", "{paragraph}")
])

para = input("Give your paragraph: ")

final_prompt = prompt.invoke(
    {
        "paragraph": para,
        "format_instructions": parser.get_format_instructions()
    }
)

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)

print(movie_data)