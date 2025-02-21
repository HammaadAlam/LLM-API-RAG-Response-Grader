import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Load Environment
load_dotenv('.env')

def gpt_rag_answers():
    """
    Extracts the first 500 answerable questions from the SQuAD2.0 Dev Set v2.0,
    queries ChromaDB for the top 5 relevant context chunks, and uses OpenAI's batch
    processing to generate answers for all the answerable questions and save the responses
    to a JSON file.
    """
    # Establish OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    print(f"Extracted {len(questions)} questions.")

    # Establish client
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

    # System prompt for the model
    system_prompt = "You are an intelligent AI. Answer all questions concisely and to the best of your ability."
    user_prompt = "Answer this question concisely and to the best of your ability: {question}"

    # Create batch request file
    tasks = []
    for id_num, question in enumerate(questions):

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

        # Create task for batch processing
        task = {
            "custom_id": f"question_{id_num + 1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        tasks.append(task)

    # Write the batch tasks to a JSONL file
    with open("gpt-RAG-answers-input-batch.jsonl", 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    # Upload the batch file to OpenAI
    batch_file = client.files.create(
        file=open("gpt-RAG-answers-input-batch.jsonl", 'rb'),
        purpose='batch'
    )

    # Run the batch using the completions endpoint
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # loop until the status of our batch is completed
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Done processing batch.")
    print(batch_job)
    print("Writing data...")
    print(check)

    # Write the results to a local file
    result = client.files.content(check.output_file_id).content
    output_file_name = "gpt-RAG-answers-output-batch.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

    # Load the output file, extract each sample output, and append to a list
    results = []
    with open(output_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)

    # Show the responses
    for item in results:
        print("Model's Response:")
        print('\t', item['response']['body']['choices'][0]['message']['content'])


# Run the function
gpt_rag_answers()