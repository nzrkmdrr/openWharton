#!/usr/bin/env python
# coding: utf-8

# In[98]:


import os
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import DocumentSummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core import PromptTemplate


import argparse


from moviepy.editor import *
from pydub import AudioSegment


import subprocess

from trulens_eval import Tru



os.environ["OPENAI_API_KEY"] = "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"

Settings.llm = OpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.2)




def convert_pptx_to_pdf_with_soffice(pptx_path, output_dir):
    """
    Converts a PPTX file to PDF using LibreOffice's command-line interface.
    
    Parameters:
    pptx_path: str - The full path to the PPTX file.
    output_dir: str - The directory where the converted PDF should be saved.
    """
    try:
        subprocess.run([
            "soffice",
            "--convert-to", "pdf:writer_pdf_Export",
            pptx_path,
            "--outdir", output_dir
        ], check=True)
        print("Conversion successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

def find_and_convert_ppt(directory):
    """
    Recursively find all PPT and PPTX files in the given directory and convert them to PDF.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.ppt','.pptx')):
                full_path = os.path.join(root, file)
                pdf_path = full_path.rsplit('.', 1)[0] + '.pdf'
                print(f"Converting {full_path} to {pdf_path}")
                convert_pptx_to_pdf_with_soffice(full_path, pdf_path)
                print(f"Conversion complete: {pdf_path}")
                print(f"Removing: {full_path}")
                os.remove(full_path)
                print(f"Remove complete: {full_path}")




def split_mp3(input_file, max_file_size_mb=20):
    """
    Splits an MP3 file into multiple segments, each smaller than a specified size.
    
    Parameters:
    input_file (str): Path to the input MP3 file.
    target_directory (str): Directory to save the split MP3 files.
    max_file_size_mb (int): Maximum size of each split file in MB. Default is 20MB.
    """
    # Load the MP3 file
    max_duration_ms = 5600000

    audio = AudioSegment.from_mp3(input_file)

    if(len(audio)<= max_duration_ms):
        return
    
    # Calculate the target duration for each split
    # Assuming average bitrate for calculation. Adjust the bitrate as per your file.
    
    # Split and export
    start = 0
    part = 1
    print(len(audio))
    while start < len(audio):
        # Calculate end time, ensuring it's not beyond the file length
        end = min(start + max_duration_ms, len(audio))
        
        # Extract part of the audio
        split_audio = audio[start:end]
        
        # Generate split filename
        split_filename = os.path.join(input_file.rsplit('.', 1)[0] + f"_part_{part}.mp3")
        
        # Export the split audio file
        split_audio.export(split_filename, format="mp3", bitrate="12k")
        
        print(f"Exported {split_filename}")
        
        # Prepare for the next iteration
        start = end
        part += 1
        
    os.remove(input_file)



#convert mp4 file to compressed mp3 (ready to be trascribe)
def convert_mp4_to_mp3(mp4_path, mp3_path):
    
    video = VideoFileClip(mp4_path)
    video.audio.write_audiofile(mp3_path)

def find_and_convert_mp4(directory):
    """
    Recursively find all PPT and PPTX files in the given directory and convert them to PDF.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                try:
                    full_path = os.path.join(root, file)
                    mp3_path = full_path.rsplit('.', 1)[0] + '.mp3'
                    print(f"Converting {full_path} to {mp3_path}")
                    convert_mp4_to_mp3(full_path, mp3_path)
                    print(f"Conversion complete: {mp3_path}")
                    print(f"Removing: {full_path}")
                    os.remove(full_path)
                    print(f"Remove complete: {full_path}")
                    sound = AudioSegment.from_file(mp3_path)
                    sound.export(mp3_path, format="mp3", bitrate="12k")
                    split_mp3(mp3_path)
                except Exception as e:
                    print(e)


#use openAI whisper to transcribe an mp3 file 
def transcribe_mp3_to_txt(mp3_path, text_file_path):
    client = OpenAI()

    audio_file = open(mp3_path, "rb")
    transcription = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file, 
      response_format="text"
    )

    with open(text_file_path, "w") as text_file:
        text_file.write(transcription)

def find_and_transcribe_mp3(directory):
    """
    Recursively find all PPT and PPTX files in the given directory and convert them to PDF.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                try:
                    full_path = os.path.join(root, file)
                    text_file_path = full_path.rsplit('.', 1)[0] + '.txt'
                    print(f"Converting {full_path} to {text_file_path}")
                    transcribe_mp3_to_txt(full_path, text_file_path)
                    print(f"Conversion complete: {text_file_path}")
                    print(f"Removing: {full_path}")
                    os.remove(full_path)
                    print(f"Remove complete: {full_path}")
                except Exception as e:
                    print(e)





# #test data code 
# documents = SimpleDirectoryReader("wharton_data", recursive = True).load_data()
# index = VectorStoreIndex.from_documents(documents)

# query_engine = index.as_query_engine()

# response = query_engine.query("how to scale yourself ")
# response.response




def main():

    parser = argparse.ArgumentParser(description="parser for LLM parameters")

    # Add arguments
    parser.add_argument('--file', '-f', type=str, help='Path to the directory',default = "wharton_data")
    parser.add_argument('--num_document', type=int, help='A number of document to for RAG to search', default = 100 )
    parser.add_argument('--chunk_size', type=int, help='chunk size of RAG', default = 1024)

    args = parser.parse_args()


    print("initiaizing database")
    directory = args.file
    num_document = args.num_document
    chunk_size = args.chunk_size

    print("converting ppt to pdf")
    find_and_convert_ppt(directory)
    print("converting mp4 to mp3")
    find_and_convert_mp4(directory)
    print("transcribing mp3 to txt")
    find_and_transcribe_mp3(directory)

        




    # load documents
    print("loading documents")
    documents = SimpleDirectoryReader(directory, recursive = True).load_data()


    # initialize settings (set chunk size)
    Settings.chunk_size = chunk_size 
    nodes = Settings.node_parser.get_nodes_from_documents(documents)



    # initialize storage context (by default it's in-memory)
    # storage_context = StorageContext.from_defaults()
    # storage_context.docstore.add_documents(nodes)



    # summary_index = SummaryIndex(nodes, storage_context=storage_context)
    # vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    print("converting documents to Summary Index")
    summary_index = SummaryIndex(nodes, recursive = True)
    # summary_index = DocumentSummaryIndex(nodes, recursive = True, show_progress= True)
    print("converting documents to Vector index")
    vector_index = VectorStoreIndex(nodes, recursive = True, show_progress= True)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=num_document,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(streaming = True)

    print("creating vector query engine")
    vector_query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    new_summary_tmpl_str = (
            "Context information from multiple sources is below. This context information are composed of business school slides, books and lecture. \n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information, previous chat history, and not prior knowledge, "
            "answer the query in the style of a business school professor\n"
            "Previous chat history: \n"
            "{query_str}\n"
            " "
            "Answer: "
        )
    
    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    vector_query_engine.update_prompts(
    {"response_synthesizer:summary_template": new_summary_tmpl}
    )


    list_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async= False,
        streaming = True,
    )

    print("creating summary query engine")
    list_tool = QueryEngineTool.from_defaults(
        query_engine=list_query_engine,
        description=(
            "Useful for summarization questions related to Wharton Classes"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from Wharton Classes"
        ),
    )

    print("configuring Agent")
    query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[
            list_tool,
            vector_tool,
        ],
    )


    #start getting user input
    previous_answer = ""
    previous_question = ""
    historical_prompt = ""
    while True:
        user_input = input("\n\nEnter something (type 'exit' to quit or 'new' to start a new chat): \n")
        if user_input.lower() == 'exit':
            print("\nExiting the program. Goodbye!\n")
            break
        if user_input.lower() == 'new':
            previous_answer = ""
            previous_question = ""
            historical_prompt = ""
            continue

        # prompt = """System: Create a system designed to answer questions from users based on the business school class content. 
        # The system accesses a detailed content database and accepts user questions through text input. When generating answers,
        #   utilize only the class content, ensuring responses are clear, concise, and directly relevant. If a question exceeds the scope 
        #   of the class content, guide the user on where to find further information. Include examples in responses for clarity and improved understanding.
        #   This is the user input:\n\n{user_input}"""

        prompt = historical_prompt+ f"Query:{user_input}\n"

        # print(f"the prompt is: {prompt}")

        
        response_stream = vector_query_engine.query(prompt)
        response_stream.print_response_stream()

        
        previous_question= str(user_input) 
        previous_answer = str(response_stream)
 
        historical_prompt += (

            f"Previous Question: {previous_question}\n"
            f"Previous Answer:{previous_answer}\n"

        )
        
        # print(f"\n\nhistorical_prompt:{historical_prompt}\n\n")
              

if __name__ == "__main__":
    main()









