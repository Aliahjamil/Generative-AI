import json
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="CineSage",
    page_icon="🎬",
    layout="centered"
)

# UI styling
st.markdown("""
<style>
.stApp {
    background-color: #f5f6fc;
    color: #1f2937;
}
.stApp p, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #1f2937;
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
.title {
    text-align: center;
    color: #5145cd !important;
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #6b7280 !important;
    margin-bottom: 30px;
}

/* Text area: visible typed text */
.stTextArea textarea,
[data-baseweb="textarea"],
[data-baseweb="base-input"] {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    -webkit-text-fill-color: #1f2937 !important;
}
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
    -webkit-text-fill-color: #9ca3af !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #1f2937 !important;
}

/* JSON code block */
[data-testid="stCode"] pre {
    background-color: #f3f4f6 !important;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.5;
}

.stButton > button {
    width: 100%;
    background-color: #5145cd;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #4035a8;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🎬 CineSage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Your Movie Information Extraction Assistant</div>',
    unsafe_allow_html=True
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1000
)

# Schema for movie information
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str

# Parser for movie information
parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
Extract movie information from the paragraph.
Use only facts provided in the paragraph.
Use null for missing optional values.
Return valid JSON matching this format:
{format_instructions}
"""),
    ("human", "{paragraph}")
])

# Input area
st.subheader("Enter Movie Paragraph")

para = st.text_area(
    "Paste your movie description below:",
    height=220,
    placeholder="Example: Interstellar is a 2014 science fiction film directed by Christopher Nolan..."
)

# Extract button
if st.button("✨ Extract Movie Information"):
    if not para.strip():
        st.warning("Please enter a movie paragraph.")
    else:
        try:
            with st.spinner("Analyzing movie information..."):
                final_prompt = prompt.invoke({
                    "paragraph": para,
                    "format_instructions": parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)
                movie_data = parser.parse(response.content)

            st.success("Movie information extracted successfully!")

            st.subheader("📽️ Extracted Information")
            st.markdown(f"**Movie Title:** {movie_data.title}")
            st.markdown(f"**Release Year:** {movie_data.release_year or 'Not mentioned'}")
            st.markdown(
                f"**Genre:** {', '.join(movie_data.genre) if movie_data.genre else 'Not mentioned'}"
            )
            st.markdown(f"**Director:** {movie_data.director or 'Not mentioned'}")
            st.markdown(
                f"**Cast:** {', '.join(movie_data.cast) if movie_data.cast else 'Not mentioned'}"
            )
            st.markdown(
                f"**Rating:** {movie_data.rating if movie_data.rating is not None else 'Not mentioned'}"
            )

            st.markdown("### Summary")
            st.write(movie_data.summary)

            with st.expander("View structured JSON"):
                st.code(
                    json.dumps(movie_data.model_dump(), indent=2, ensure_ascii=False),
                    language="json"
                )

        except Exception as e:
            st.error(f"Unable to extract movie information: {e}")

##Interstellar is a 2014 science fiction film directed by Christopher Nolan and written by Jonathan Nolan and Christopher Nolan. The movie stars Matthew McConaughey as Cooper, Anne Hathaway as Amelia Brand, Jessica Chastain as Murph, and Michael Caine as Professor Brand. The story follows a former NASA pilot named Cooper who travels through a wormhole near Saturn with a team of astronauts to find a new habitable planet as Earth faces environmental collapse. The film explores themes of love, sacrifice, time, gravity, and human survival. Hans Zimmer composed the powerful and emotional soundtrack. Interstellar received widespread critical acclaim and has an IMDb rating of approximately 8.7 out of 10.

##3 Idiots is a 2009 Indian coming-of-age comedy-drama directed by Rajkumar Hirani. The film stars Aamir Khan, R. Madhavan, Sharman Joshi, Kareena Kapoor, and Boman Irani. Set in an engineering college, the story explores themes of friendship, academic pressure, and following one's passion rather than societal expectations. The movie became one of the highest-grossing Indian films of its time and received immense praise for its humor and emotional depth. It has a rating of 8.4 on IMDb and remains a cultural favorite across generations.