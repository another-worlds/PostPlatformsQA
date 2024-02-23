from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
# t = SystemMessagePromptTemplate.from_template("rfrf")
# print(t.prompt.template)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.retrievers import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain import hub



from langchain_community.vectorstores.chroma import Chroma

from langchain.document_loaders.pdf import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from prompt_helper import template
# markdown/sentence splitters
# ParentDocumentRetriever - cascade split chunks
# metadata?

load_dotenv()
docs_path = "docs_cut.pdf"
docs_test_path = "pdf_one_pager.pdf"

embeddings  = OpenAIEmbeddings()

def docs_to_chroma(docs:str):
    loader = PyPDFLoader(docs_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(loader.load())
    db = Chroma.from_documents(docs, OpenAIEmbeddings())
    return db

def docs_to_vectorDB(docs: str) -> FAISS:
    loader = PyPDFLoader(docs_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(loader.load())
    db = FAISS.from_documents(docs, embeddings)
    return db

def format_context(docs):
    context = []    
    for doc in docs:
        page_content = doc.page_content.replace("\n", " ")
        page_number = doc.metadata.get('page')
        context.append({"page_content": page_content,
                        "page_number" : page_number})
    
    context = [str(entry) for entry in context]
    formatted_context = "\n".join(context)
    print( 
f"""
----------------------------------
{formatted_context}
----------------------------------
""")
    return formatted_context



def db_to_retrieval_chain(db: FAISS, query:str, k:int = 6):# -> LLMChain:
    docs = db.similarity_search(query,k)
    context = format_context(docs)
    prompt_template = PromptTemplate.from_template(template)
    llm = OpenAI(temperature=0.1)
    chain = prompt_template | llm
    return chain.invoke({"query":query, "context":context})


metadata_field_info = [
    AttributeInfo(
        name="page_content",
        description="The fragment from the postplatforms whitepaper",
        type='string'
    ),
    AttributeInfo(
        name="page_number",
        description="The number of the page the fragment sources from",
        type='integer'
    )
]

mode = 'agent'
if mode == 'test':
    db = docs_to_vectorDB(docs_path) #change to docs_path
elif mode == 'selfQuery' or 'agent':
    db = docs_to_chroma(docs_path)


print(
"""
----------------------------------
ENTERING CHAIN
----------------------------------
""")

#

if __name__ == "__main__":
    
    while True:
        user_input = input()
        if mode == 'test':
            answer = db_to_retrieval_chain(db, user_input)
            print(answer)
        elif mode == 'selfQuery' or 'agent':    
            llm = OpenAI(temperature=0.1)     
            retriever = SelfQueryRetriever.from_llm(
                llm =llm,
                vectorstore = db,
                document_contents = "Fragments from the postplatforms project.",
                metadata_field_info=metadata_field_info,
                verbose=True
            )
            if mode =='agent':
                tool = create_retriever_tool(
                    retriever,
                    'Postplatforms docs search',
                    'Postplatforms whitepaper search tool to extract relevant context')
                tools = [tool]
                # Get the prompt to use - you can modify this!
                prompt = hub.pull("hwchase17/openai-functions-agent")
                prompt.messages[0].template=template
                print(prompt)
            elif mode =='selfQuery':
                docs = retriever.invoke(user_input)
                prompt_template = PromptTemplate.from_template(template)
                chain = prompt_template | llm
                answer = chain.invoke({"query":user_input, "context":format_context(docs)})
                print(answer)
