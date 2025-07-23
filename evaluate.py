import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from Database.Chroma import build_rag_index
from model.llm import llm  

# indexing the PDF
vector_store, chunks = build_rag_index(
    pdf="D:/PROJECT-PAPER/10ms/Dataset/HSC26-Bangla1st-Paper.pdf",
    collection_name="10ms_collection",
    persist_directory="./chroma_langchain_db",
    chunk_size=200,
    chunk_overlap=50,
)

# Retrieval + threshold
def retrieve_with_threshold(
    query: str,
    k: int = 5,
    threshold: float = 0.3
) -> List[Tuple[Document, np.ndarray, float]]:
    """
    Returns up to k Documents whose cosine similarity with the query >= threshold,
    along with their raw embedding and the similarity score.
    """
    # embed the query
    q_emb = np.array(vector_store._embedding_function.embed_query(query)).reshape(1, -1)

    # fetch top-k raw from Chroma (docs + embeddings only)
    resp = vector_store._collection.get(
        include=["documents", "embeddings"],
        limit=k
    )
    docs  = resp["documents"]
    embs  = resp["embeddings"]

    results = []
    for doc_text, emb in zip(docs, embs):
        emb_arr = np.array(emb).reshape(1, -1)
        score = float(cosine_similarity(q_emb, emb_arr)[0,0])
        if score >= threshold:
            doc = Document(page_content=doc_text, metadata={})
            results.append((doc, emb_arr, score))

    # sort by descending score
    results.sort(key=lambda x: x[2], reverse=True)
    return results

# Bengali prompt template
template = """\
আপনি একজন বাংলা ভাষী সহকারী।
নিম্নলিখিত প্রসঙ্গ ব্যবহার করে *শুধুমাত্র* প্রশ্নের উত্তর দিন—
কোনো নতুন তথ্য তৈরি করবেন না।
যদি প্রসঙ্গে উত্তর না থাকে, “দুঃখিত, আমার কাছে যথাযথ তথ্য নেই” লিখুন।

প্রসঙ্গ:
{context}

প্রশ্ন: {question}

উত্তর:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# embedding function for evaluation
sbert = HuggingFaceEmbeddings(
    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
    model_kwargs={"device": "cuda"} 
)

def evaluate_rag(test_set: List[Tuple[str, str]]) -> List[dict]:
    results = []
    for question, gold in test_set:
        # retrieve relevant documents
        retrieved = retrieve_with_threshold(question, k=5, threshold=0.0)
        docs, embs, retrieval_cosines = zip(*retrieved) if retrieved else ([], [], [])
        
        context = "\n\n".join(d.page_content for d in docs)
        prompt_str = prompt.format(context=context, question=question)
        pred = llm.invoke(prompt_str).strip()   

        # groundness
        ctx_tokens = set(context.split())
        pred_tokens = pred.split()
        grounded = sum(t in ctx_tokens for t in pred_tokens)
        groundness = grounded / max(len(pred_tokens), 1)

        # relevance
        gold_vec = np.array(sbert.embed_query(gold)).reshape(1, -1)
        pred_vec = np.array(sbert.embed_query(pred)).reshape(1, -1)
        relevance = float(cosine_similarity(gold_vec, pred_vec)[0,0])

        results.append({
            "question": question,
            "gold_answer": gold,
            "pred_answer": pred,
            "retrieval_cosines": [round(c,3) for c in retrieval_cosines],
            "groundness": round(groundness,3),
            "relevance": round(relevance,3),
        })
    return results


if __name__ == "__main__":
    test_set = [
        ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "ষোড়শ বছর"),
        
    ]
    metrics = evaluate_rag(test_set)
    for m in metrics:
        print("Q:", m["question"])
        print(" Pred:", m["pred_answer"])
        print(" Gold:", m["gold_answer"])
        print(" Retrieval cosines:", m["retrieval_cosines"])
        print(" Groundness:", m["groundness"])
        print(" Relevance:", m["relevance"])
        print("-" * 50)
