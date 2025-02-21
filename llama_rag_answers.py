import os
import json
import time

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load Environment
load_dotenv('.env')

def llama_rag_answers():

    """
    Extracts the first 500 answerable questions from the SQuAD2.0 Dev Set v2.0
    and uses Azure to run Llama 3.2 11B Vision Instruct to serially generate
    answers to all the answerable questions and save the responses to a json file.
    """

    # Establish Client
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
    )

    # Establish ChromaDB Client
    chroma_client = chromadb.PersistentClient(path="../data/my_chromadb")

    # Initialize embedding functions to generate vector representations of the text
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # Get the documents collection with the provided embedding function
    collection = chroma_client.get_collection(
        name="squad2.0_contexts",
        embedding_function=openai_ef
    )

    # Load SQuAD2.0 dataset
    with open('dev-v2.0.json') as f:
        data = json.load(f)

    questions = []
    # Extract the first 500 answerable questions
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa["is_impossible"]:  # Skip unanswerable questions
                    questions.append(qa["question"])
                if len(questions) >= 500:
                    break
            if len(questions) >= 500:
                break
        if len(questions) >= 500:
            break

    # Open the output file in append mode
    with open('llama-RAG-answers.jsonl', 'a') as output_file:
        for i, question in enumerate(questions, 1):
            # Query ChromaDB to get the 5 most relevant context chunks
            search_results = collection.query(
                query_texts=[question],
                n_results=5
            )

            # Get the chunks of context
            context_chunks = [doc for doc_list in search_results['documents'] for doc in doc_list]

            # Combine the question and context chunks to form the input prompt
            context = "\n".join(context_chunks)
            prompt = f"""You are a helpful AI assistant that answers questions using data returned by a search engine.

            Guidelines:
            1. You will be provided with a question by the user, you must answer that question, and nothing else.
            2. Your answer should come directly from the provided context from the search engine.
            3. Do not make up any information not provided in the context.
            4. If the provided question does not contain the answers, respond with 'I am sorry, but I am unable to answer that question.'
            5. Be aware that some chunks in the context may be irrelevant, incomplete, and/or poorly formatted.

            Here is the provided context: {context}

            Here is the question: {question}

            Your response:"""

            # Submit the question and context to the Llama model
            response = client.complete(
                messages=[
                    SystemMessage(
                        content=(
                            "You are an intelligent AI. Answer all questions concisely and to the best of your ability."
                        )
                    ),
                    UserMessage(content=prompt),
                ],
            )

            # Structure the result
            result = {
                "question": question,
                "response": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }

            print(f"{i} / {len(questions)} questions answered: {question}")
            output_file.write(json.dumps(result) + '\n')

            time.sleep(1)

    print("Completed answering all questions.")
    print("Results saved in llama-RAG-answers.jsonl")

    # print the results
    print("Model's Response:")
    print('\t', response.choices[0].message.content)
    print()
    print(f"Input Tokens:  {response.usage.prompt_tokens}")
    print(f"Output Tokens: {response.usage.completion_tokens}")
    print(f"Cost: ${response.usage.prompt_tokens * 0.0003 / 1000 + response.usage.completion_tokens * 0.00061 / 1000}")

llama_rag_answers()