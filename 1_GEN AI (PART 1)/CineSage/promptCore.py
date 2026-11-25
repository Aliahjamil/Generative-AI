from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=300
)


# Instantiation using from_template (recommended)

from langchain_core.prompts import ChatPromptTemplate

movie_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are CineSage, an intelligent movie information
        extraction and summarization assistant.

        Extract useful movie information from the user's paragraph.

        Identify the following details:

        1. Movie title
        2. Release year
        3. Genre
        4. Main cast and character names, if mentioned
        5. Director, writer, and composer, if mentioned
        6. Plot and storyline
        7. Setting and main conflict
        8. Central themes and key topics
        9. IMDb rating and other ratings, if mentioned
        10. Awards and recognition, if mentioned
        11. Soundtrack and notable music
        12. Critical and audience reception
        13. Distinctive features and why the movie is notable
        14. A quick summary in 2-3 sentences

        Rules:
        - Extract only facts supported by the paragraph.
        - Never invent information.
        - Write "Not mentioned" for missing details.
        - If the movie title cannot be identified confidently,
          write "Not identified".
        - Clearly label inferred genres.
        - Use headings and concise bullet points.
        - Avoid unnecessary repetition.
        - Do not return JSON.
        """
    ),
    (
        "human",
        "Extract useful information from this movie paragraph:\n\n{paragraph}"
    )
])

para=input("Give your paragrapgh:")

final_prompt = movie_prompt.format_messages(paragraph=para)

response = model.invoke(final_prompt)

print(response.content)
