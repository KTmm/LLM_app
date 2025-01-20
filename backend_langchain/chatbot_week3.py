### This script is extract key experiments and summary from paper A and validate if they can be reproduced or validated in paper B
import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Name of the existing Pinecone index that contains BOTH papers (A and B)
index_name = 'research-paper-store-1'
index = pc.Index(index_name)

# Describe the index stats—useful for debugging.
index_stats = index.describe_index_stats()
print("Index stats:", index_stats)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='text-embedding-ada-002'
)

target_paper_vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="MingL_NC_2022.pdf"
)

validate_paper_vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="JanC_Science_2022.pdf"  # Optional, if you used a namespace
)

target_paper_retriever = target_paper_vectorstore.as_retriever(search_kwargs={"k": 30})
validate_paper_retriever = validate_paper_vectorstore.as_retriever(search_kwargs={"k": 30})

docs = target_paper_retriever.get_relevant_documents("What does paper A say about the mannose-6-phosphate pathway?")
for d in docs:
    print("--- Retrieved Doc ---")
    print("Metadata:", d.metadata)
    print("Page Content:", d.page_content[:300])

llm = ChatOpenAI(
    model_name='gpt-4',
    temperature=0.0  
)

qa_extract_experiments = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",     
    retriever=target_paper_retriever
)

qa_validation = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",      
    retriever=validate_paper_retriever
)

summary_query = (
    "Please summarize MingL_NC_2022.pdf paper key experiments, including details "
    "of the experimental process and the main results as well as the conclusions."
)

# summary_query = (
#     "Please provide a concise summary of Paper A's key experiments, "
#     "covering the experimental process, main results, and any notable conclusions. "
#     "Focus on the mannose-6-phosphate (M6P) pathway if mentioned."
# )


summary = qa_extract_experiments.run(summary_query)
print("=== Summary of Paper A's Key Experiments ===")
print(summary)
print("============================================\n")

validation_query = (
    f"Here are the key experiments from MingL_NC_2022.pdf:\n\n{summary}\n\n"
    "Does Jan_C_Science_2022.pdf reproduce or validate these experiments? Provide evidence or references."
)

validation_response = qa_validation.run(validation_query)

print("=== Does Paper B Reproduce/Validate Paper A's Key Experiments? ===")
print(validation_response)
print("==================================================================")