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

# (Optional) Describe the index stats—useful for debugging.
index_stats = index.describe_index_stats()
print("Index stats:", index_stats)


embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='text-embedding-ada-002'
)

# 3. Wrap the Pinecone index in a LangChain VectorStore
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

# 4. Create a retriever
# By default, it will search among all stored chunks; you can also add filters
target_paper_retriever = target_paper_vectorstore.as_retriever(search_kwargs={"k": 5})
validate_paper_retriever = validate_paper_vectorstore.as_retriever(search_kwargs={"k": 5})


# 3. Create the LLM
llm = ChatOpenAI(
    model_name='gpt-4',
    temperature=0.0  
)

# 7. STEP A: Extract "key experiments" from Paper A
#    We'll create a RetrievalQA chain that focuses on Paper A content.
qa_extract_experiments = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",     
    retriever=target_paper_retriever
)

qa_validation = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",      # "stuff" => simplest chain type
    retriever=validate_paper_retriever
)

summary_query = (
    "Please summarize MingL_NC_2022.pdf paper key experiments, including details "
    "of the experimental process and the main results as well as the conclusions."
)

summary = qa_extract_experiments.run(summary_query)
print("=== Summary of Paper A's Key Experiments ===")
print(summary)
print("============================================\n")

# Construct a query that specifically asks for key experiments from "Paper A"
validation_query = (
    f"Here are the key experiments from MingL_NC_2022.pdf:\n\n{summary}\n\n"
    "Does Jan_C_Science_2022.pdf reproduce or validate these experiments? Provide evidence or references."
)

validation_response = qa_validation.run(validation_query)

print("=== Does Paper B Reproduce/Validate Paper A's Key Experiments? ===")
print(validation_response)
print("==================================================================")