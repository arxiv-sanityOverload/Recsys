import os
import sys
import time
import shutil
import multiprocessing
from subprocess import call
import pdfx
import json

# Gobal DIR
txt_path = "text/"                         # "Change to according to your directory"
ref_path = "references/"
pdf_path = "pdf/"

def pdf_dir():
    data = []
    for paths, dirs, file in os.walk(pdf_path):
        for f in file:
            data.append((paths, f))
    return data

have = set(os.listdir(txt_path))


def pdf_extract(dirs):
    '''
    Function takes filename and path to the file as a tuple and save the extracted text and references from PDF file to txt_path   
    dirs = ("pdf_data/", "filename.pdf")

    '''
    paths, filename = dirs
    file = filename.replace(".pdf", ".txt")
    file_json = filename.replace(".pdf", ".json")
    if file in have:
        print("file alreafy extracted!!")
    elif filename == ".DS_Store":
        pass
    else:
        print("read pdf file", filename)
        cmd_text_extractor = "pdfx %s -t -o %s" % (os.path.join(paths, filename), txt_path+file)
        pdf = pdfx.PDFx(os.path.join(paths, filename))
        references_dict = pdf.get_references_as_dict()
        print("extrated reference of:", file)
        os.system(cmd_text_extractor)
        print("extracted pdf_file:", file)
        with open(ref_path+file_json, 'w') as fp:
            json.dump(references_dict, fp)
        print("save json to reference:", file_json)
        time.sleep(0.01)

pool = multiprocessing.Pool()
if __name__ =='__main__':
    filenames = pdf_dir()
    filenames = filenames[0:10]
    try:
        result = pool.map(pdf_extract, filenames)
        pool.close()
        pool.join()
    except Exception as e:
        print(e)
        pass