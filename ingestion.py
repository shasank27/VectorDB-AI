import os 
import dotenv
dotenv.load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == "__main__":
    print("IntroToVectorDB")
    document  = TextLoader("mediumblog.txt").load()
    splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap=100)
    texts = splitter.split_documents(document)

    # print(texts)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # print(embeddings)
    PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ['INDEX_NAME'])