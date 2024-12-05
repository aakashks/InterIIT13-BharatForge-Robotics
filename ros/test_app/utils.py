from PIL import Image, ImageDraw
from icecream import ic

def get_count_from_coord_data(coord_data):
    """
    Get the total count of objects from the given coordinate data.
    """
    count = 0
    for obj_id in coord_data:
        obj_data = coord_data[obj_id]
        count += len(obj_data["points"])
        
    return count
        

def get_topk_imgs_from_coord_data(coord_data, k=4):
    """
    Retrieve the top-k object images from the given coordinate data.
    Draws a red point at the specified coordinates.
    """
    # Extract all (object, path) pairs from nested structure
    path_pairs = []
    for obj_id in coord_data:
        obj_data = coord_data[obj_id]
        object_name = obj_data["object"]
        # Add all paths for this object
        for point_data in obj_data["points"]:
            img = Image.open(point_data["image_path"])
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions to scale coordinates
            width, height = img.size
            
            # Draw red dots at each coordinate pair
            for x, y in zip(point_data["x_coordinates"], point_data["y_coordinates"]):
                # Convert percentage to actual pixels
                pixel_x = int(x * width / 100)
                pixel_y = int(y * height / 100)
                
                # Draw red circle with radius 5
                draw.ellipse(
                    [(pixel_x-5, pixel_y-5), (pixel_x+5, pixel_y+5)],
                    fill='red',
                    outline='red'
                )
            
            path_pairs.append((object_name, img))
            
    # Return only top k pairs (they're already sorted by confidence)
    return path_pairs[:k]


def get_topk_paths_from_coord_data(coord_data, k=4):
    """
    Retrieve the top-k object paths from the given coordinate data.
    """
    # Extract all (object, path) pairs from nested structure
    path_pairs = []
    for obj_id in coord_data:
        obj_data = coord_data[obj_id]
        object_name = obj_data["object"]
        # Add all paths for this object
        for point_data in obj_data["points"]:
            path_pairs.append((object_name, point_data["image_path"]))
            
    # Return only top k pairs (they're already sorted by confidence)
    return path_pairs[:k]


# def test_get_topk_paths_from_coord_data():
#     """
#     Test function for get_topk_paths_from_coord_data.
#     """
#     # Define some sample coordinate data
#     coord_data = {
#         0: {
#             "object": "bed",
#             "points": [
#                 {
#                     "x_coordinates": [49.8],
#                     "y_coordinates": [32.3],
#                     "image_path": "/home/user1/s_ws/images/10.png"
#                 },
#                 {
#                     "x_coordinates": [48.9],
#                     "y_coordinates": [35.4],
#                     "image_path": "/home/user1/s_ws/images/11.png"
#                 }
#             ]
#         },
#         1: {
#             "object": "dustbin",
#             "points": [
#                 {
#                     "x_coordinates": [37.9],
#                     "y_coordinates": [47.3],
#                     "image_path": "/home/user1/s_ws/images/7.png"
#                 },
#                 {
#                     "x_coordinates": [11.9],
#                     "y_coordinates": [50.3],
#                     "image_path": "/home/user1/s_ws/images/6.png"
#                 },
#                 {
#                     "x_coordinates": [25.6],
#                     "y_coordinates": [55.3],
#                     "image_path": "/home/user1/s_ws/images/8.png"
#                 }
#             ]
#         }
#     }

#     # Call the function
#     result = get_topk_paths_from_coord_data(coord_data, k=8)
#     ic(result)
    
# if __name__ == "__main__":
#     test_get_topk_paths_from_coord_data()