import os 
import dotenv
dotenv.load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

if __name__ == "__main__":
    print("Retrieval")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    llm = ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash")
    
    query = "What is Vector?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)
    
    vectorstore = PineconeVectorStore(embedding= embeddings, index_name = os.environ['INDEX_NAME'])

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain = combine_docs_chain)

    result = retrieval_chain.invoke(input={"input": query})
    print(result)