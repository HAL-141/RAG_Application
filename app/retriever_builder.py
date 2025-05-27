from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

# RRF
def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
) -> list[str]:
    # 各ドキュメントのコンテンツ（文字列）とそのスコアの対応を保持する辞書を準備
    content_score_mapping = {}

    # 検索クエリごとにループ
    for docs in retriever_outputs:
        # 検索結果のドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content

            # 初めて登場したコンテンツの場合はスコアを0で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0
            # (1 / (順位 + k)) のスコアを加算
            content_score_mapping[content] += 1 / (rank + k)

    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]


# hybrid_retriever作成
def build_hybrid_retriever(
    docs: list[Document],
    embeddings: Embeddings,
    persist_dir: str = "../chroma_db"
) -> Runnable:
    # ドキュメント分割 (トークン数対策)
    midpoint = len(docs) // 2
    docs_part1 = docs[:midpoint]
    docs_part2 = docs[midpoint:]
    
    # Chromaベクターストアを準備   
    # 1回目：ベクトルストア初期化＋保存
    chroma_db = Chroma.from_documents(
        documents=docs_part1,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    # 2回目：既存ベクトルストアに追記（同じ persist_directory に接続）
    chroma_db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    chroma_db.add_documents(docs_part2)
    
    # chroma_retriever
    chroma_retriever = chroma_db.as_retriever()
    chroma_retriever = chroma_retriever.with_config(
        {"run_name": "chroma_retriever"}
    )
    # BM25_retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever = bm25_retriever.with_config(
        {"run_name": "bm25_retriever"}
    )
    
    # hybrid_retriever
    hybrid_retriever = (
        RunnableParallel({
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever,
        })
        | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
        | reciprocal_rank_fusion
    )

    return hybrid_retriever