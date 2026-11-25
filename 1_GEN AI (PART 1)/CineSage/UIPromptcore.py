import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CineSage | Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #f5f6fc;
    font-family: 'Inter', sans-serif;
}

/* Force readable text colors regardless of Streamlit theme */
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #252747;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 1.5rem 0 2rem 0;
}

.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #20234b;
    margin-bottom: 0.4rem;
}

.hero p {
    color: #737895;
    font-size: 1rem;
}

.input-label {
    color: #20234b;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.stTextArea textarea {
    background: #ffffff;
    border: 1px solid #dfe2f2;
    border-radius: 14px;
    color: #252747;
    font-size: 0.95rem;
    line-height: 1.7;
}

.stTextArea textarea:focus {
    border-color: #6855e8;
    box-shadow: 0 0 0 1px #6855e8;
}

.stButton > button {
    width: 100%;
    min-height: 3.1rem;
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 700;
    font-size: 1rem;
    background: linear-gradient(135deg, #5145cd, #8565f5);
    transition: 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border: none;
    filter: brightness(1.08);
}

/* Button text stays white */
.stButton > button p,
.stButton > button:hover p {
    color: white;
}

/* Result card: style the real container so content sits inside it */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #e2e5f3 !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 8px 25px rgba(42, 48, 100, 0.05);
}

.result-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #20234b;
    margin-bottom: 0.75rem;
}

.footer {
    text-align: center;
    color: #8a8eaa;
    font-size: 0.8rem;
    margin-top: 2.5rem;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
    <h1>🎬 CineSage</h1>
    <p>Your intelligent movie information extractor</p>
</div>
""", unsafe_allow_html=True)


# Initialize model
@st.cache_resource
def load_model():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=800
    )


# Chat prompt template
movie_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are CineSage, an intelligent movie information
        extraction and summarization assistant.

        Extract the following information from the user's paragraph:

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
        - If the title cannot be identified confidently,
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

# Input section
st.markdown(
    '<div class="input-label">📝 Movie Paragraph</div>',
    unsafe_allow_html=True
)

paragraph = st.text_area(
    "Movie paragraph",
    placeholder="Paste your movie description here...",
    height=220,
    label_visibility="collapsed"
)

# Extraction button
if st.button("✨ Extract Movie Information", use_container_width=True):
    if not paragraph.strip():
        st.warning("Please enter a movie paragraph first.")
    else:
        try:
            with st.spinner("Analyzing the movie paragraph..."):
                model = load_model()
                messages = movie_prompt.format_messages(paragraph=paragraph)
                response = model.invoke(messages)

            with st.container(border=True):
                st.markdown(
                    '<div class="result-title">🎞️ Extracted Movie Information</div>',
                    unsafe_allow_html=True
                )
                st.markdown(response.content)

        except Exception as e:
            st.error(f"Unable to extract information: {e}")

# Footer
st.markdown("""
<div class="footer">
    CineSage · Movie Information Extractor
</div>
""", unsafe_allow_html=True)

###Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Michael Caine. The story revolves around a group of astronauts who travel through a wormhole near Saturn in search of a new home for humanity as Earth faces environmental collapse. The movie was widely appreciated for its emotional depth, scientific accuracy, and Hans Zimmer's powerful soundtrack. It holds a rating of 8.6 on IMDb and is often considered one of the greatest sci-fi films of the 21st century.

###The Dark Knight, released in 2008, is a superhero crime thriller directed by Christopher Nolan and produced by Warner Bros. The film features Christian Bale as Batman alongside Heath Ledger, Aaron Eckhart, and Gary Oldman. The plot focuses on Batman's battle against the chaotic criminal mastermind known as the Joker, whose actions plunge Gotham City into anarchy. Heath Ledger's performance earned widespread acclaim and a posthumous Academy Award. The film currently holds an IMDb rating of 9.0 and is praised for its intense storytelling and grounded realism.