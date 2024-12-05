from icecream import ic
from vlm import run_multiple_image_query_same_prompt
import asyncio
import re

def run_clip_on_objects(object_list, collection, topk=5):
    prompt = [f'a photo of a {obj}' for obj in object_list]

    results = collection.query(
        query_texts=prompt,
        n_results=topk,
    )
    ic(results)

    object_and_path = {}
    for i, obj in enumerate(object_list):
        image_paths = [d['image_path'] for d in results['metadatas'][i]]
        object_and_path[i] = {'object': object_list[i], 'image_paths': image_paths}
        
    ic(object_and_path)
    
    return object_and_path


def run_vlm(object_and_path, concurrent_requests=25, timeout=120):
    results = {}
    for i, dic in object_and_path.items():
        template = f"Point to the {dic['object']} in the image."
        image_paths = dic['image_paths']
        ic(template)
        ic(image_paths)
        result = asyncio.run(run_multiple_image_query_same_prompt(image_paths, template, timeout=timeout, concurrent_requests=concurrent_requests))
        ic(result)
        results[i] = {'object': dic['object'], 'points': []}
        
        for r in result:
            points = extract_points(r)
            if points:
                results[i]['points'].append(points)
    
    return results


def extract_points(text):
    # Parse the <points> tag and extract relevant data
    match = re.search(r'<points([^>]*)>(.*?)</points>', text)
    if not match:
        return None

    attributes = match.group(1)
    main_message = match.group(2)

    # Extract the coordinates
    x_coords = []
    y_coords = []
    alt_message = None

    # Parse the attributes of the points tag
    for attr in attributes.split():
        if attr.startswith('x'):
            x_coords.append(float(attr.split('=')[1].strip('"')))
        elif attr.startswith('y'):
            y_coords.append(float(attr.split('=')[1].strip('"')))
        elif attr.startswith('alt'):
            alt_message = re.search(r'alt="([^"]*)"', attributes).group(1)

    return {
        "x_coordinates": x_coords,
        "y_coordinates": y_coords,
        "alt_message": alt_message,
        "main_message": main_message
    }


def return_coordinates(results):
    return results