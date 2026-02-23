import sys
sys.path.append("/home/bispl_02/hangeol/HypothesisRAG")
try:
    from retriever import create_retriever
    retriever = create_retriever("mirage", "MedCPT", "Textbooks")
    docs, _ = retriever.retrieve("What is a headache?", k=1)
    if not docs:
        print("FAILED: No docs")
    else:
        print("SUCCESS:", docs[0].get("title", "No Title"))
except Exception as e:
    print("ERROR:", e)
