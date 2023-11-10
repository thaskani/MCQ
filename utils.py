
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import openai
from langchain.llms import OpenAI
import re
import json

def read_pdf(pdf):

    # For multiple files
    content = ''
    if len(pdf):
        for i in range(len(pdf)):
            pdf_reader = PdfReader(pdf[i])
            for page in range(len(pdf_reader.pages)):
                content += pdf_reader.pages[page].extract_text()
    # For single file
    else:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content += page.extract_text()
    
    return content

def text_splitter(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = splitter.split_text(text)
    return chunks

def vectore_store(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embedding=embeddings)
    return knowledge_base

def semantic_serch(vectoreStore,query):
    relevant_docs = vectoreStore.similarity_search(query,k=2) 
    return relevant_docs


def get_answer(query,relevant_docs):
      
    llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10, "max_length":512},)
    # llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=relevant_docs, question=query)
    return response


def create_mcq(answer):
    response_schemas = [
        ResponseSchema(name="question", description="Question generated from provided input text data."),
        ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma separated."),
        ResponseSchema(name="answer", description="Correct answer for the asked question.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    format_instructions = output_parser.get_format_instructions()

    # create ChatGPT object
    chat_model = ChatOpenAI()
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""When a text input is given by the user, 
            please generate multiple choice questions from it along with the correct answer. 
            \n{format_instructions}\n{user_prompt}""")  
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )
    final_query = prompt.format_prompt(user_prompt = answer)
    final_query_output = chat_model(final_query.to_messages())
    markdown_text = final_query_output.content
    json_string = re.search(r'{(.*?)}', markdown_text, re.DOTALL).group(1)
    print(json_string)
    return json_string