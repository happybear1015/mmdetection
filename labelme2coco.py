# images and json files, all in a dir.
import json
import os
import numpy as np
from PIL import Image
from skimage.draw import polygon as draw_polygon
from pycocotools import mask as mask_utils

def convert_labelme_poly_to_coco(labelme_json_dir, output_coco_path):
    # 创建COCO格式的字典
    coco_data = {
        "images": [],
        "categories": [],
        "annotations": []
    }
    # 添加类别信息（如果有多个标注文件，可以根据需要进行合并或自定义）
    categories = [
        {"id": 0, "name": "NX"},
        {"id": 1, "name": "M"},
        # 添加更多类别...
    ]

    # 添加类别信息
    for category in categories:
        coco_category = {
            'id': category['id'],
            'name': category['name']
        }
        coco_data['categories'].append(coco_category)

    image_id = 1
    annotation_id = 1

    # 处理每个图像和对应的标注文件
    for json_file in os.listdir(labelme_json_dir):
        if json_file.endswith('.json'):
            labelme_json_path = os.path.join(labelme_json_dir, json_file)

            # 加载LabelMe的JSON文件
            with open(labelme_json_path, 'r') as f:
                labelme_data = json.load(f)

            # 提取图像信息
            image_file = labelme_data['imagePath']
            image_path = os.path.join(labelme_json_dir, image_file)
            image = Image.open(image_path)
            image_width, image_height = image.size

            # 添加图像信息
            coco_image = {
                'id': image_id,
                'file_name': image_file,
                'width': image_width,
                'height': image_height
            }
            coco_data['images'].append(coco_image)

            # 提取标注信息
            annotations = labelme_data['shapes']
            for annotation in annotations:
                category_id = 0  # 默认为0，表示car类别
                polygon = annotation['points']  # 多边形边界点信息

                # 构建多边形的掩膜
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                x = [int(round(p[0])) for p in polygon]
                y = [int(round(p[1])) for p in polygon]

                # 将坐标限制在图像范围内
                x = np.clip(x, 0, image_width - 1)
                y = np.clip(y, 0, image_height - 1)

                rr, cc = draw_polygon(y, x)
                mask[rr, cc] = 1

                # 将掩膜编码为RLE格式
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')  # 将字节转换为字符串

                # 将多边形的坐标取整
                polygon = [[int(round(p[0])), int(round(p[1]))] for p in polygon]

                # 添加标注信息
                coco_annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'segmentation': [np.asarray(polygon).flatten().tolist()],
                    'area': float(mask_utils.area(rle)),  # 计算掩膜区域面积
                    'bbox': mask_utils.toBbox(rle).tolist(),
                    'iscrowd': 0  # 默认为非crowd
                }
                coco_data['annotations'].append(coco_annotation)

                annotation_id += 1

            image_id += 1

    # 保存为COCO格式的JSON文件
    with open(output_coco_path, 'w') as f:
        json.dump(coco_data, f)

# 示例用法
# labelme_json_dir = r'C:\Users\15135\Downloads\PaddleYOLO-release-2.6\dataset\coco\train2017'
# output_coco_path = r'C:\Users\15135\Downloads\PaddleYOLO-release-2.6\dataset\coco\annotations\instances_train2017.json'

labelme_json_dir = r'C:\Users\15135\Downloads\PaddleYOLO-release-2.6\dataset\coco\val2017'
output_coco_path = r'C:\Users\15135\Downloads\PaddleYOLO-release-2.6\dataset\coco\annotations\instances_val2017.json'
convert_labelme_poly_to_coco(labelme_json_dir, output_coco_path)

