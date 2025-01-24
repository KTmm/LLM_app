### This script is extract key experiments and summary from paper A and validate if they can be reproduced or validated in paper B
import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from .models import db, ChatMessage

def compare_paper(target_paper, validate_paper):
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
        namespace=target_paper
        )

    validate_paper_vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=validate_paper  # Optional, if you used a namespace
    )

    target_paper_retriever = target_paper_vectorstore.as_retriever(search_kwargs={"k": 30})
    validate_paper_retriever = validate_paper_vectorstore.as_retriever(search_kwargs={"k": 30})

    # docs = target_paper_retriever.get_relevant_documents("What does paper A say about the mannose-6-phosphate pathway?")
    # for d in docs:
    #     print("--- Retrieved Doc ---")
    #     print("Metadata:", d.metadata)
    #     print("Page Content:", d.page_content[:300])

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
        "Please summarize {target_paper} paper key experiments, including details "
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
        f"Here are the key experiments from {target_paper}:\n\n{summary}\n\n"
        "Does {validate_paper} reproduce or validate these experiments? Provide evidence or references."
    )

    validation_response = qa_validation.run(validation_query)
    print("=== Does Paper B Reproduce/Validate Paper A's Key Experiments? ===")
    print(validation_response)
    print("==================================================================")

    # retrieval_chain = (
    #     {
    #         "context": retriever.with_config(run_name="Docs"),
    #         "question": RunnablePassthrough(),
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    return validation_response

def call_chat(target, validate):
    answer = ""
    for chunk in compare_paper(target, validate):
        answer += chunk
        yield {"token": chunk}

    chat_message = ChatMessage(user_id=1, question=target, answer=answer)
    db.session.add(chat_message)
    db.session.commit()

