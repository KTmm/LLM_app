### the script embedded 4 pdf format papers and upsert into pinecone
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from openai import OpenAI

new_index_name = "research-paper-store-1"

def get_all_pdf_names(directory_path):
    pdf_file_list = [file for file in os.listdir(directory_path) if file.lower().endswith('.pdf')]
    print(pdf_file_list)
    return pdf_file_list

# Extract text from a PDF
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [str(doc) for doc in documents]
    return texts

# Define a function to create embeddings
def create_embeddings(texts):
    client = OpenAI()
    embeddings_list = []
    for text in texts:
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        embeddings_list.append(response.data[0].embedding)
    return embeddings_list

def create_embedding_vectors(embedding_list, texts, paper_id):
    embedding_vector = []

    for i, embedding in enumerate(embedding_list):
        vector_id = f"{paper_id}_chunk_{i}"
        chunk_text = texts[i]
    
        metadata = {
        "paper_id": paper_id,
        "chunk_index": i,
        "text": chunk_text
        }
    
        embedding_vector.append((vector_id, embedding, metadata))
    return embedding_vector    

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if new_index_name not in pc.list_indexes():
    pc.create_index(
        name=new_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
    )
)

index = pc.Index(new_index_name)

paper_directory_path = "research_papers"  
pdf_list = get_all_pdf_names(paper_directory_path)
for pdf_name in pdf_list:
    file_path = paper_directory_path + "/" + pdf_name
    texts = process_pdf(file_path)
    embeddings = create_embeddings(texts)
    vectors_to_upsert = create_embedding_vectors(embeddings, texts, pdf_name)
    index.upsert(vectors=vectors_to_upsert, namespace=pdf_name)
