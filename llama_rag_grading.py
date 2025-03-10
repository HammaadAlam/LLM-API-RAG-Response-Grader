import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


def llama_rag_grading():
    """
    Uses the OpenAI API to run GPT-4o in batch mode in order to
    score the responses of LLama as either true or false and
    save the outputs in a json file providing reasoning.
    """
    # Load Environment
    load_dotenv('.env')
    with open('dev-v2.0.json') as f:
        correct_data = json.load(f)

    # Load Llama's responses from the output file
    with open("llama-RAG-answers.jsonl", 'r') as f:
        llama_data = [json.loads(line) for line in f if line.strip()]

    llama_answers = []  # Define list of llama answers
    for data in llama_data:
        question = data["question"]
        response_content = data["response"]
        llama_answers.append({
            "question": question,
            "response": response_content
        })

    correct_answers = [] # Define list of correct answers
    for article in correct_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible'] and qa['answers']:
                    answer = qa['answers'][0]['text'].lower()
                    correct_answers.append(answer)
                if len(correct_answers) >= 500:
                    break
            if len(correct_answers) >= 500:
                break
        if len(correct_answers) >= 500:
            break

    # Define JSON schema for the structured output to the json
    json_schema = {
        "name": "grading_output",
        "schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "A short explanation of why the student's answer was correct or incorrect."
                },
                "score": {
                    "type": "boolean",
                    "description": "True if the student's answer is correct, false otherwise."
                }
            },
            "required": ["explanation", "score"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Define the grading prompts
    system_prompt = """You are a teacher tasked with determining whether a student’s answer to a question was correct, based on a set of possible correct answers. You must only use the provided possible correct answers to determine if the student’s response was correct."""
    user_prompt = """Question: {question}
    Student's Response: {student_response}
    Possible Correct Answers: {correct_answer}

    Your response should be a valid JSON in the following format:
    {{
    "explanation": "(str): A short explanation of why the student’s answer was correct or incorrect.",
    "score": "(bool): true if the student’s answer was correct, false if it was incorrect."
    }}"""

    tasks = []  # Create grading tasks for each question and response pair
    for idx, (correct_answer, llama_answer) in enumerate(zip(correct_answers, llama_answers), 1):
        if correct_answer is None:
            continue
        question = llama_answer['question']
        student_response = llama_answer['response']
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question,
                                                           student_response=student_response,
                                                           correct_answer=correct_answer)}
        ]

        custom_id = f"{idx}. {question}"

        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": messages,
            }
        }
        tasks.append(task)

    # Write the batch tasks to a JSONL file.
    input_filename = "llama-RAG-grading-input-batch.jsonl"
    with open(input_filename, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # Initialize the OpenAI client.
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    # Upload the batch file to OpenAI
    batch_file = client.files.create(
        file=open(input_filename, 'rb'),
        purpose='batch'
    )

    # Create a batch job for grading
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print("Waiting for batch job to complete...")
    # Poll the batch job status until completion
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Batch processing completed.")

    # Retrieve and save the grading results
    result = client.files.content(check.output_file_id).content
    output_file_name = "llama-RAG-2-19-2025-hw4.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

    print(f"Grading results saved to {output_file_name}.")


llama_rag_grading()