import time
import asyncio
import requests
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, APITimeoutError
from .prompts import return_prompt

load_dotenv()

sem = asyncio.Semaphore(20)
SENTINEL = object()

model_name="meta-llama/Llama-3.3-70B-Instruct"

# Openai Client setup that can be configured to run onprem or to openai api compatible endpoint
client = AsyncOpenAI(
    base_url="https://model.url/v1",
    api_key="api-key",
    timeout=600.0
)

def process_results(column_value, cases, index, column_name):
    if column_value:
        # check if csv file exists
        column_value = column_value.split(', ')
        cases.at[index, column_name] = column_value

def save_results(cases, index, csv_path):
    if index % 100 == 0 or index == len(cases) - 1:
        cases.to_csv(csv_path, index=False, encoding="utf-8")

def infer_model(model_name: str, use_openai_api: bool=False, prompt: list=None):
    if use_openai_api:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.6,
            messages=[
                {"role": "user", "content": '\n'.join(prompt)}
            ]
        )
        result = completion.choices[0].message.content
    else:
        pload = {
            "messages": [
                {"role": "user", "content": '\n'.join(prompt)}
            ],
            "model": model_name,
            "top_p": 0.95,
            "min_p": 0.0,
            "presence_penalty": 1.0,
            "top_k": 20,
            "temperature": 0.6,
            "max_tokens": 8000,
        }
        response = requests.post(
            str(client.base_url) + "chat/completions",
            json=pload,
            headers={"Authorization": f"Bearer {client.api_key}"}
        )
        response.raise_for_status()
        response_dict = response.json()
        result = response_dict['choices'][0]['message']['content']
        reasoning = response_dict['choices'][0]['message']['reasoning_content']
    return result, reasoning

async def generate_icd_code(task: str, doc: str, loc_codes: list, cases: pd.DataFrame, index: int, csv_path: str, model_name: str=model_name, retry_count: int=3, use_openai_api: bool=False):
    prompt = return_prompt(task, doc, loc_codes)
    if task == "icd_10":
        pred_column = 'Generated_ICD-10'
        reasoning_column = 'Generated_ICD-10_reasoning'
    elif task == "icd_10_wo_locs":
        pred_column = 'Generated_ICD-10_wo_locs'
        reasoning_column = 'Generated_ICD-10_wo_locs_reasoning'
    elif task == "icd_o":
        pred_column = 'Generated_ICD-O'
        reasoning_column = 'Generated_ICD-O_reasoning'    
    else:
        raise ValueError(f"Unknown task: {task}")    
    async with sem:
        for attempt in range(retry_count):
            try:
                result, reasoning = infer_model(model_name, use_openai_api, prompt)
                if reasoning:
                    process_results(reasoning, cases, index, column_name=reasoning_column)
                process_results(result, cases, index, column_name=pred_column)
                save_results(cases, index, csv_path)
                return result
            
            except APITimeoutError as e:
                print(f"Timeout occurred (attempt {attempt+1}/{retry_count})")
                if attempt < retry_count - 1:
                    # Add exponential backoff for retries
                    wait_time = (2 ** attempt) * 1  # 1, 2, 4, 8... seconds
                    print(f"Waiting {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed after {retry_count} attempts")
                    return None  # Return None after all retries have failed