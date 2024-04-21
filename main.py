from typing import Union
import time
from fastapi import FastAPI, File, UploadFile
from model.predict import predict
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension


@app.post("/predict/")
async def upload_file(image: UploadFile = File()):
    extension = get_file_extension(image.filename)
    contents = await image.read()
    unix_time = int(time.time())
    file_name = f'''{unix_time}{extension}'''
    with open(f'''./tmp/{file_name}''' ,"wb") as f:
        f.write(contents)

    results = predict(file_name)
    os.remove(f'''./tmp/{file_name}''')
    return results

app.mount("/result", StaticFiles(directory="result"), name="result")