
from pathlib import Path
import cv2
import os
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat


from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)

from docling.document_converter import DocumentConverter, PdfFormatOption


class UNDocumentParser:
    def __init__(self, doc_path, output_path, device='gpu'):
        self.doc_path = doc_path
        self.output_path = output_path 
        self.device = device

        if self.device == 'cpu':
            accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CPU
            )
   
        if self.device == 'gpu':
            accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA
            )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
    
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
            
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


    def get_all_lang(self):
        return [f.name.split(".pdf")[0][-2:] for f in Path(self.doc_path).iterdir() if f.suffix.lower() == ".pdf" and f.is_file()]

    def convert_document(self, folder_path, file_name):
        print(f"Converting document: {file_name}")
        try:
            converted_doc = self.converter.convert(self.doc_path + folder_path + '/' + file_name)
            export_markdown = converted_doc.document.export_to_markdown()
            return export_markdown
        except Exception as e:
            print(f"Error: {e}")
            return None
    
   
    def markdown_to_json(self, exported_markdown):
       
        lines = exported_markdown.split("\n")
        structured_data = []
        
        list_buffer = []
        in_list = False

        for line in lines:
            line = line.strip()

            if not line:
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if heading_match:
                if in_list:
                    structured_data.append({"type": "list", "items": list_buffer})
                    list_buffer = []
                    in_list = False
                structured_data.append({
                    "type": "heading",
                    "level": len(heading_match.group(1)),
                    "text": heading_match.group(2)
                })
                continue

            list_match = re.match(r"^[-*] (.+)", line)
            ordered_list_match = re.match(r"^\d+\.\s+(.+)", line)

            if list_match or ordered_list_match:
                list_item = list_match.group(1) if list_match else ordered_list_match.group(1)
                list_buffer.append(list_item)
                in_list = True
                continue

            
            if in_list:
                structured_data.append({"type": "list", "items": list_buffer})
                list_buffer = []
                in_list = False
            structured_data.append({"type": "paragraph", "text": line})

        if in_list:
            structured_data.append({"type": "list", "items": list_buffer})

        return structured_data

    
    def find_header_slope(self, image):
        """Detect lines in the header to find the angle for skew correction."""
        header_height = image.shape[0] // 10
        header_image = image[:header_height]

        gray = cv2.cvtColor(header_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=250, maxLineGap=50)

        angles = []
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                return median_angle
        except TypeError:
            return 0

    def deskew_image(self, image, angle):
        """Rotate the image to correct skew based on the angle."""
        #print(f"Rotating image by {angle} degrees")
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def detect_two_columns(self, folder_name, pdf_name, debug=False):

        images = convert_from_path(self.doc_path + folder_name + '/' + pdf_name, first_page=1, last_page=1, dpi=300)
        img = np.array(images[0])
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
            
        vertical_white_space = np.sum(binary, axis=0)
        
        # Focus on the specified range of pixel columns
        column_start = 500
        column_end = 2000
        restricted_white_space = vertical_white_space[column_start:column_end]
        
        # Threshold for low white space in pixel columns
        low_white_space_threshold = 10000  # Adjust this threshold if necessary
        low_white_space_columns = restricted_white_space < low_white_space_threshold

        # Check for a continuous span of low white space columns
        min_continuous_span = 50  # Minimum number of consecutive columns to qualify as a "gap"
        current_span = 0
        for is_low in low_white_space_columns:
            if is_low:
                current_span += 1
                if current_span >= min_continuous_span:
                    if debug:
                        # Debug visualization
                        plt.figure(figsize=(15, 5))
                        plt.plot(range(column_start, column_end), restricted_white_space, label='White Space Distribution')
                        plt.axhline(low_white_space_threshold, color='red', linestyle='--', label='Threshold')
                        plt.title(f'{pdf_name}')
                        plt.xlabel('Pixel Column')
                        plt.ylabel('White Space Amount')
                        plt.legend()
                        plt.pause(0.001)

                        plt.show()
                    return True
            else:
                current_span = 0  

        if debug:
            # Debug visualization
            plt.figure(figsize=(15, 5))
            plt.plot(range(column_start, column_end), restricted_white_space, label='White Space Distribution')
            plt.axhline(low_white_space_threshold, color='red', linestyle='--', label='Threshold')
            plt.title(f'{pdf_name}')
            plt.xlabel('Pixel Column')
            plt.ylabel('White Space Amount')
            plt.legend()
            plt.pause(0.001)
            plt.show()
        
        return False
    
    def check_format(self, doc_folder):
        file_names = [f.name for f in Path(self.doc_path + doc_folder).iterdir() if f.suffix.lower() == ".pdf" and f.is_file()]

        vote = {"single-column": 0, "double-column": 0}

        for file_name in file_names:
            result = self.detect_two_columns(doc_folder, file_name)

            if result:
                vote["double-column"] += 1
            else:
                vote["single-column"] += 1

        if vote["double-column"] > vote["single-column"]:
            return "double-column"
        else:
            return "single-column"
        

    def parse_pdf(self):

        all_doc_folders = [name for name in os.listdir(self.doc_path) if os.path.isdir(os.path.join(self.doc_path, name))]

        for doc_folder in tqdm(all_doc_folders):
            # format = self.check_format(doc_folder)
            # print(f"{doc_folder}: {format} detected")
            print(f"Parsing {doc_folder} started.")
            if not os.path.exists(self.output_path + doc_folder):
                os.mkdir(self.output_path + doc_folder)
            
                all_file_names = [f.name for f in Path(self.doc_path + doc_folder).iterdir() if f.suffix.lower() == ".pdf" and f.is_file()]
                for file_name in all_file_names:
                    if file_name.split('.pdf')[0][-2:] not in ['zh', 'ar', 'ru']:
                        converted_doc = self.convert_document(doc_folder, file_name)
                        if converted_doc:
                            exported_json = self.markdown_to_json(converted_doc)
                            with open(self.output_path + doc_folder + '/' + file_name.split('.pdf')[0] + '-parsed.json' , "w", encoding="utf-8") as f:
                                json.dump(exported_json, f, indent=4, ensure_ascii=False)
                        else:
                            print(f"Error parsing {file_name}")
                            continue
                    else:
                        continue
                print(f"Parsing {doc_folder} finished.")
                print("===============================")
           

if __name__ == "__main__":
    parser = UNDocumentParser(doc_path="/srv/liri_storage/data/yingqiang/projects/spirituality/crawled_data/", output_path='/srv/liri_storage/data/yingqiang/projects/spirituality/parsed_data/', device='gpu')
    
    parser.parse_pdf()
