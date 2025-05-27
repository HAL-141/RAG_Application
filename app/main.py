from env_loader import load_keys
from rag_chain import build_rag_chain

def main():
    #環境変数を読み込む
    load_keys()
    
    hybrid_rag_chain = build_rag_chain()
    result = hybrid_rag_chain.invoke({"question":"メガ損保３社の近年の業績はどうですか"})
    print(result)
    

if __name__ == "__main__":
    main()