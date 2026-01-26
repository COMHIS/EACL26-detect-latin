import asyncio
from openai import AsyncOpenAI
import time
import json
from tqdm.asyncio import tqdm
from pathlib import Path
import base64
import argparse
import os
import mimetypes
import httpx


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_as_data_uri(image_path):
    # Guess MIME type
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")
    
    # Encode image as base64
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    # Return full data URI
    return f"data:{mime_type};base64,{base64_image}"


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def write_json(data, file_path, indent=4, print_log=True):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


async def async_openai_predict(client, model, query, qa_item, image_url=None, min_pixels=512*28*28, max_pixels=2048*28*28):
    """
    Asynchronous prediction function that handles a single request.
    """
    page_anno = qa_item['data'] if 'data' in qa_item else qa_item
    try:
        if not image_url:
            content = [{
                'type': 'text',
                'text': query,
            }]
        else:
            content = [{
                'type': 'text',
                'text': query,
            }, {
                'type': 'image_url',
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                'image_url': {
                    'url': image_url,
                }
            }]
        response = await client.chat.completions.create(
            model=model,
            messages=[{
                'role': 'user',
                'content': content
            }],
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 15000
            },
            temperature=0,
            seed=2025,
            max_tokens=20000,
        )
        pred = response.choices[0].message.content
        # Return a dictionary containing the original info and the prediction result
        return {
            "image_url": page_anno['imageUrl'],
            "prompt": query,
            "model_pred": pred,
            "error": None
        }
    except Exception as e:
        error_msg = f"API error for {page_anno['imageUrl']}: {e}"
        return {
            "image_url": page_anno['imageUrl'],
            "prompt": query,
            "model_pred": "",
            "error": error_msg
        }


async def main(args):
    """
    The entire main logic with generalized paths
    """
    client = AsyncOpenAI(
        base_url=f'http://localhost:{args.port}/v1/',
        api_key='EMPTY',
        http_client=httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_connections=1024
            )
        )
    )

    data_dir = Path(args.data_path)
    qas = read_json(data_dir)
    pred_dir = Path(args.output_dir) / f'{args.model_name}_{args.test_name}.json'

    # Prepare all tasks
    tasks = []
    for qa in qas:
        page_anno = qa['data'] if 'data' in qa else qa
        img_path = Path(args.image_dir) / f"{page_anno['eccoId']}_{page_anno['pageNumber']}.jpg"
        img_inp = encode_image_as_data_uri(img_path) if 'V' in args.modality else None
        
        if 'pageTextClean' in qa and not args.ori_ocr:
            page_text = qa['pageTextClean']
        else:
            page_text = qa['pageText']

        if 'L' in args.modality:
            query = f'''{args.prompt}

            OCR Text: {page_text}
            '''
        else:
            query = args.prompt

        task = async_openai_predict(client, args.model_name, query, page_anno, image_url=img_inp)
        tasks.append(task)

    # Execute all tasks concurrently
    print(f"Processing {len(tasks)} requests concurrently...")
    preds = await tqdm.gather(*tasks)
    
    print(f"All {len(preds)} requests completed.")

    # Save results
    dir_path = os.path.dirname(pred_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    write_json(preds, pred_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference script with specified model and test name")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate')
    parser.add_argument('--test_name', type=str, required=True, help='Test name or tag for this evaluation run')
    parser.add_argument('--modality', type=str, required=True, help='Input modality for this evaluation run: L for text only, V for image only, VL for both')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to use for the model, no page text included')
    parser.add_argument('--max_len', type=int, required=False, default=4096, help='The maximum output token length')
    parser.add_argument('--ori_ocr', action='store_true', help='Whether to use original OCR text')
    parser.add_argument('--port', type=int, required=False, default=8000, help='The port number for the server')
    parser.add_argument('--data_path', type=str, required=False,
                        default='data/latin_annotation_final.json',
                        help='Path to input data file')
    parser.add_argument('--image_dir', type=str, required=False,
                        default='data/images',
                        help='Path for images')
    parser.add_argument('--output_dir', type=str, required=False,
                        default='output',
                        help='Path for output')

    args = parser.parse_args()

    start_time = time.time()
    asyncio.run(main(args))
    end_time = time.time()
    
    print('==================')
    print(f"Total execution time: {end_time - start_time:.2f} seconds")