# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate


def generate_prompts():
    prompt_template = """
Based on the query, you should give some suggestions related to health care for the following QUESTION. 
Don't write down too much and write some similar questions in the end, just give me some short suggestions. 
Don't try to make up an answer, if you don't know just say that you don't know.
Use only the following pieces of context to answer the question at the end.


{context}
SPLIT_END_MARKER!!!
Question: {question}
Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return PROMPT
