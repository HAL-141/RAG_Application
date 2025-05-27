from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Pydanticで出力形式を規定
class Answer(BaseModel):
    question: str = Field(description="summary of the question")
    answer: str = Field(description="summary of the answer")
    explanation: str = Field(description="explanation of the answer")
    keywords: list[str] = Field(description="keywords helpful for explaining the question")

# OutputParser & 出力フォーマットの作成
output_parser = PydanticOutputParser(pydantic_object=Answer)
format_instructions = output_parser.get_format_instructions()

# システムプロンプトテンプレート
system_prompt = f"""\
あなたは保険業界についての質問に対する回答者です。
以下の条件を厳守し、参考情報を元に、必ず**JSON形式で**回答してください。

# 条件 :
1. 以下の４項目をすべて含めること :
    - question: 質問の要点の要約
    - answer: 回答の要約
    - explanation: 回答の背景・根拠、具体的説明（400字以内）
    - keywords: 回答に関連する重要語句のリスト
2. 出力全体を与えられたformat_instrcutionsに従って書くこと

# 参考情報 :
{{context}}

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}\n\n 回答は必ずJSON形式で、指定されたキーをすべて含めてください。")
]).partial(format_instructions=format_instructions)