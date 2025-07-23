import sys
sys.stdout.reconfigure(encoding="utf-8")
from langgraph.graph import START, StateGraph
from Database.Chroma import build_rag_index
from model.llm import llm
from Preprocess.PDFprocess import PDFprocessor
from langchain.schema import HumanMessage
from transformers import AutoTokenizer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# load once at top
vector_store,chunks = build_rag_index(
    pdf="D:/PROJECT-PAPER/10ms/Dataset/HSC26-Bangla1st-Paper.pdf",
    collection_name="10ms_collection",
    persist_directory="./chroma_langchain_db",
    chunk_size=2500,
    chunk_overlap=100,
)


template = """\
আপনি একজন বাংলা ভাষী সহকারী।
নিম্নলিখিত প্রসঙ্গ ব্যবহার করে *শুধুমাত্র* প্রশ্নের উত্তর দিন—  
কোনো নতুন তথ্য তৈরি করবেন না।  
যদি প্রসঙ্গে উত্তর না থাকে, “১) দুঃখিত, আমার কাছে যথাযথ তথ্য নেই” লিখুন।  

প্রসঙ্গ:
{context}

প্রশ্ন: {question}

উত্তর:
"""

prompt = PromptTemplate(
    input_variables=["context","question"],
    template=template
)

# Build the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",                             
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt},
)

#inspect retrieval during dev
def debug_retrieval(q):
    docs = vector_store.similarity_search(q, k=5)
    print("\n--- Retrieved Chunks ---")
    for i, d in enumerate(docs, 1):
        print(f"[{i}]\n{d.page_content[:200]!r}\n")
    print("------------------------\n")

def ask_loop():
    print("RAG is Ready — Type 'exit' to leave\n")
    while True:
        q = input("Question: ").strip()
        if not q or q.lower() in ("exit","quit"):
            break

        debug_retrieval(q)   

        ans = qa.run(q)
        print("\n Answer:", ans.strip(), "\n" + "-"*60 + "\n")

if __name__ == "__main__":
    ask_loop()
