import os
import shutil
import stat
import time
import subprocess
import tempfile
import gc
import pdfplumber
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
    
# 読み取り専用ファイルを削除する補助関数
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# post-checkoutフックのセキュリティ制限を回避し、安全にGitリポジトリをcloneする関数
def safe_git_clone(clone_url: str, branch: str = "main", target_dir: str = None) -> str:
    
    # デフォルトは一時ディレクトリ
    if target_dir is None:
        target_dir = os.path.join(tempfile.gettempdir(), "temp_repo")

    # 既存ディレクトリを削除（リソース開放してから削除）
    if os.path.exists(target_dir):
        gc.collect()         # ガーベジコレクションでリソース開放
        time.sleep(1)        # 少し待ってから削除
        try:
            shutil.rmtree(target_dir, onerror=remove_readonly)
        except Exception as e:
            raise RuntimeError(f"既存の {target_dir} を削除できませんでした: {e}")

    # クローン（checkout はあとで行う）
    result = subprocess.run(
        ["git", "clone", "--no-checkout", clone_url, target_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git clone failed:\n{result.stderr}")

    # post-checkout フックを削除（セキュリティ制限対策）
    hook_path = os.path.join(target_dir, ".git", "hooks", "post-checkout")
    if os.path.exists(hook_path):
        try:
            os.remove(hook_path)
        except Exception as e:
            raise RuntimeError(f"post-checkout フックの削除に失敗しました: {e}")

    # 明示的に checkout
    result = subprocess.run(
        ["git", "checkout", branch],
        cwd=target_dir,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git checkout failed:\n{result.stderr}")

    return target_dir

# pdfplumber でPDFを読み込む
# PyPDFLoaderだと決算資料等に使用されるUniJIS-UTF16-Hをうまく読み込めないため
# 「CropBox missing from /Page, defaulting to MediaBox」のログがページ数分出力されるが抑制できなかったため放置　
# ※動作には問題なし
def load_documents(clone_url: str, branch: str = "main") -> list[Document]:
    
    pdf_file_paths = []

    # 指定パスのpdfファイルのみを抽出する。
    def file_filter(file_path: str) -> list[str]:
        if file_path.endswith(".pdf") and "Insurance_documents" in file_path:
            pdf_file_paths.append(file_path)
            return True
        return False

    # クローン実行
    repo_path = safe_git_clone(clone_url, branch)
    # Gitから対象ファイルパスを取得
    loader = GitLoader(
        clone_url=clone_url,
        repo_path=repo_path,
        branch=branch,
        file_filter=file_filter,
    )
    loader.load()  # Git clone + フィルタリングしたファイルパスをpdf_file_pathsに追加
    print("Gitから対象ファイルのパスを取得")

    # 対象ドキュメントリストを返却
    docs = []
    print("ドキュメント　ロード中...")
    for pdf_path in pdf_file_paths:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        if full_text.strip():
            docs.append(Document(page_content=full_text, metadata={"source": pdf_path}))

    return docs