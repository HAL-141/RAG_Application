from document_loader import safe_git_clone, load_documents
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from retriever_builder import build_hybrid_retriever
from langchain_openai import ChatOpenAI
from prompt_and_schema import prompt, output_parser, format_instructions
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from operator import itemgetter
from langchain_core.documents import Document

def build_rag_chain(
    clone_url: str = "https://github.com/HAL-141/RAG_Application.git",
    branch: str = "main",
    docs: [list[Document]] = None,
    embedding_model: str = "text-embedding-3-small",
):
    if docs is None:
        #Document loader
        docs = load_documents(clone_url=clone_url, branch=branch)

    # Document transformer
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    split_docs = text_splitter.split_documents(docs)
        
    #Embedding model
    embeddings = OpenAIEmbeddings(model=embedding_model)

    #Retriever
    hybrid_retriever = build_hybrid_retriever(split_docs, embeddings)

    #Model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    hybrid_rag_chain = (
    RunnableMap({
        "question": RunnablePassthrough(),
        "context": itemgetter("question") | hybrid_retriever,
        "format_instructions": lambda _: format_instructions
    })
    | prompt | model | output_parser
    ).with_config({"run_name": "hybrid_rag_chain"})

    return hybrid_rag_chain
