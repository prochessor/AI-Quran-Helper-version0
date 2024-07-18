import requests
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence




llm = ChatGroq(
    temperature=0.7,
    model="mixtral-8x7b-32768",
    api_key="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"
)

# Function to fetch Quran data from API
def fetchChapter(chapter):
    url = f"https://api.quran.com/api/v4/verses/by_chapter/{chapter}"
    params = {
        "language": "en",
        "translations": [84],  # Translation ID for English
        "tafsirs": [169],  # Tafsir ID (e.g., for Ibn Kathir)
        "per-page": 286
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

def fetchTafsir(chapter, ayah):
    tafsirId = 169
    url = f"https://staging.quran.com/api/qdc/tafsirs/{tafsirId}/by_ayah/{chapter}:{ayah}"
    response = requests.get(url)
    data = response.json()
    return data

def truncate_text(text, max_tokens=3000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    return text

def clean_text(text):
    # Remove Arabic characters
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    text = arabic_pattern.sub(' ', text)

    # Remove HTML tags
    html_pattern = re.compile(r'<.*?>')
    text = html_pattern.sub(' ', text)

    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)

    return truncate_text(text.strip())
if __name__ == "__main__":
    splits = []
    for chapter in range(1, 2):
        # First we need to find the ayah count
        data = fetchChapter(chapter)
        ayahCount = data['pagination']['total_records']
        data = data['verses']
        for ayah in range(1,ayahCount+1):
            tafsir = fetchTafsir(chapter, ayah)
            ayah_text = data[ayah - 1]['translations'][0]['text']  # Adjusted index
            tafsir_text = clean_text(tafsir['tafsir']['text'])
            source = f"({chapter}:{ayah})"
        
            doc =Document(page_content=f"Ayah: {ayah_text} Tafsir: {tafsir_text}", metadata={"source": source})
            splits.append(doc)

    embeddings = HuggingFaceEmbeddings()
    vectorDB = FAISS.from_documents(documents=splits, embedding=embeddings)

    prompt_template = """You are my helper to understand Quran, You are provided a context and a question below, answer the question following the instructions below:
    1. Answer only using the context provided 
    2. If you don't found any answer from the context then simply say, "I cannot find any relevant information"
    3. Your answer should not have any text like "Here is relevant information...." or "According to the context......", I need direct answers
    
    CONTEXT: {context}

    QUESTION: {question}"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    sequence = RunnableSequence(PROMPT|llm )

    chain = RetrievalQAWithSourcesChain.from_llm(
        retriever=vectorDB.as_retriever(),
        llm = llm,
        question_prompt=PROMPT
        
        
        )

    result = chain.invoke({"question": "What does straight path means"}, return_only_outputs=True)

    print(result["answer"])

    print("-"*30)
    print(result["sources"])


    print(result)

    






