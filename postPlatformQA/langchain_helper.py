from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import AgentType

from langchain.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

load_dotenv()



def langchain_agent():
    llm = OpenAI(temperature=0.7)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    result = agent.run(
        "What is the average height of a dog? Multiply it by average dog age."
    )
    
    return result


embeddings = OpenAIEmbeddings()

def youtube_to_vectorDB(url_string: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(url_string)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_reponse_from_query(db: FAISS, query: str, k: int=4) -> str:
    
    similar_docs = db.similarity_search(query,k)
    similar_docs_content = " ".join([d.page_content for d in similar_docs])
    
    
    prompt_template = PromptTemplate(
        input_variables=[f"question", "similar_docs"],
        template=
        """
        You are a helpful assistant for youtube.
        
        You must answer user's question about videos 
        based on the information provided 
        from the video transcript.
        
        Here is the question: {question}
        
        Here is the relevant video transcript: {docs}
        
        Only use information from the specific documents
        assigned. If you feel you don't have enough relevant
        information from the transcript, say 
        "The video doesn't provide relevant context".
        
        Answer in 150 words.
        
        """
        )
    
    llm = OpenAI(temperature=0.7)
    question_chain = LLMChain(llm=llm,prompt=prompt_template)
    response = question_chain.run(
        question=query,
        docs=similar_docs_content)
    
    return response.replace("\n","")

if __name__ == "__main__":
    langchain_agent()