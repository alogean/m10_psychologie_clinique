from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader

def my_first_chain():
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class technical documentation writer."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    responce = chain.invoke({"input": "how can langsmith help with testing?"})
    print(responce)
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])

def my_image_pdf_loader():
    print("load pdf")
    loader = PyPDFLoader("FontaineFontaine_2006.pdf", extract_images=True)
    pages = loader.load()
    print(pages[4].page_content)

def my_pdf_loader(path_to_file):
    loader = PyPDFLoader(path_to_file)
    pages = loader.load_and_split()
    print(pages[1])

if __name__ == "__main__":
    #my_first_chain()
    my_pdf_loader("./guide-de-pratique-dépression-2015.pdf")