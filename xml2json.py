import json
import xml.etree.ElementTree as ET
import os

def convert_voc_to_coco(xml_dir, output_json):
    """
    Args:
        xml_dir: Directory containing XML annotation files
        output_json: Output path for COCO JSON file
    """
    
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    categories = {}
    category_id = 1
    image_id = 1
    annotation_id = 1

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            print(f'filename: {filename}\n'
                  f'size: {size}\n'
                  f'width: {width}\n'
                  f'height: {height}\n')

            image_info = {
                'id': image_id,
                'width': width,
                'height': height,
                'file_name': filename
            }
            coco_format['images'].append(image_info)

            # objects 처리
            for obj in root.findall('object'):
                class_name = obj.find('name').text

                # 카테고리가 처음 등장하면 추가해줌
                if class_name not in categories:
                    categories[class_name] = category_id
                    coco_format['categories'].append({
                        'id': category_id,
                        'name': class_name,
                        'supercategory': 'object'
                    })
                    category_id += 1
                
                # 바운딩 박스 변환
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # COCO 바운딩 박스 형식 [x, y, height, width]
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                area = bbox_width * bbox_height

                difficult = obj.find('difficult')
                is_crowd = int(difficult.text) if difficult is not None else 0

                # annotation 생성
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': categories[class_name],
                    'segmentation': [],
                    'area': area,
                    'bbox': [xmin, ymin, bbox_width, bbox_height],
                    'iscrowd': is_crowd
                }

                coco_format['annotations'].append(annotation)
                annotation_id += 1
            
            image_id += 1

        except Exception as e:
            print(f'Error processing {xml_file}: {str(e)}')
            continue
    print('- coco_fotmat -')
    for key, value in coco_format.items():
        print(f'{key}: {value}')
    print()
    
    # COCO JSON file로 저장
    with open(output_json, 'w') as f:
        # json.dump() → 파이썬 자료(여기선 dictionary)를 json으로 작성해줌
        # coco_format → 파이썬 자료, f → write 가능한 파일, indent → 들여쓰기정도(선택)
        json.dump(coco_format, f, indent=2)

    print(f'Conversion completed Successfully')
    print(f"Proessed {len(coco_format['images'])} images")
    print(f"Created {len(coco_format['annotations'])} annotations")
    print(f"Found {len(coco_format['categories'])} categories: {list(categories.keys())}")
    print(f'Output saved to: {output_json}')

def main():
    xml_path = 'data/'
    output_json_path = 'data/converted_voc_to_coco.json'
    convert_voc_to_coco(xml_path, output_json_path)

if __name__ == '__main__':
    main()