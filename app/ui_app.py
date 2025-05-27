import streamlit as st
from rag_chain import build_rag_chain
from env_loader import load_keys

# 環境変数を読み込み
load_keys()

# チェーン構築（初回のみ）
@st.cache_resource(show_spinner="RAGチェーン構築中...")
def load_chain():
    return build_rag_chain()

rag_chain = load_chain()

# UIレイアウト
st.title("保険業界向け RAG QAシステム")
query = st.text_input("質問を入力してください")

if st.button("回答を取得"):
    if not query.strip():
        st.warning("質問を入力してください")
    else:
        with st.spinner("回答生成中..."):
            result = rag_chain.invoke({"question": query})
            st.subheader("--回答--")
            st.write(result.answer)
            st.caption("--説明--")
            st.write(result.explanation)
            st.subheader("関連キーワード")
            hashtags = " ".join(f"#{kw}" for kw in result.keywords)
            st.markdown(hashtags)
        

