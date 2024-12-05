import asyncio
import re
from typing import Dict, List, Optional, Union

from icecream import ic
from vlm import run_multiple_image_query_same_prompt


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





def extract_points(text: str) -> Optional[Dict[str, Union[List[float], str]]]:
    """
    Extract coordinates and messages from point/points XML-like tags.
    Handles both single coordinates (x="10.5") and multiple coordinates (x1="10.5" x2="9").
    Returns all valid coordinate pairs even if some coordinates are missing.

    Args:
        text: Input text containing point/points tags

    Returns:
        Dictionary containing coordinates and messages, or None if no match
    """
    # Match either <point> or <points> tags
    pattern = r'<point(?:s)?([^>]*)>(.*?)</point(?:s)?>'
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return None

    attributes = match.group(1)
    main_message = match.group(2).strip()

    # Initialize dictionaries for coordinates
    x_dict = {}
    y_dict = {}
    alt_message = None

    try:
        # Extract x coordinates (both x="val" and x1="val", x2="val" formats)
        x_matches = re.finditer(r'x(\d*)="([^"]*)"', attributes)
        for x_match in x_matches:
            index = x_match.group(1) if x_match.group(1) else '1'
            x_dict[int(index)] = float(x_match.group(2))

        # Extract y coordinates (both y="val" and y1="val", y2="val" formats)
        y_matches = re.finditer(r'y(\d*)="([^"]*)"', attributes)
        for y_match in y_matches:
            index = y_match.group(1) if y_match.group(1) else '1'
            y_dict[int(index)] = float(y_match.group(2))

        # Extract alt message
        alt_match = re.search(r'alt="([^"]*)"', attributes)
        if alt_match:
            alt_message = alt_match.group(1)

    except ValueError as e:
        print(f"Error parsing coordinates: {e}")
        return None

    # Find valid coordinate pairs
    x_coords = []
    y_coords = []

    # Get all indices that have both x and y coordinates
    valid_indices = sorted(set(x_dict.keys()) & set(y_dict.keys()))

    for idx in valid_indices:
        x_coords.append(x_dict[idx])
        y_coords.append(y_dict[idx])

    if not x_coords or not y_coords:
        print("Error: No valid coordinate pairs found")
        return None

    return {
        "x_coordinates": x_coords,
        "y_coordinates": y_coords,
        # "alt_message": alt_message,
        # "main_message": main_message,
    }


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
        
        for j, r in enumerate(result):
            points = extract_points(r)
            if points:
                points['image_path'] = dic['image_paths'][j]
                results[i]['points'].append(points)
    
    return results