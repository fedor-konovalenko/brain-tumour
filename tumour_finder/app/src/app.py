from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import argparse
import logging
from model import evaluate
from model_resnet import predict
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

INP_SIZE = 224
DEVICE = "cpu"


# image size according to other applications


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request})


@app.post("/predict-detect")
def process_request(file: UploadFile, request: Request):
    """save file to the local folder and send the image to the process function"""
    save_pth = "tmp/" + file.filename
    app_logger.info(f'processing file - segmentation {save_pth}')
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read())
    status, result = predict(save_pth, INP_SIZE)
    if status == 'OK' and result != 'no_tumour':
        app_logger.info(f'classification result {result}')
        res, res_path = evaluate(save_pth)
        message = f"Detected {result} Tumour"
        return templates.TemplateResponse("detect_form.html",
                                          {"request": request,
                                           "res": res,
                                           "message": message, "path": res_path})
    elif result == 'OK' and result == 'no_tumour':
        app_logger.info(f'processing status {result}')
        message = f"No tumour detected"
        return templates.TemplateResponse("detect_form.html",
                                          {"request": request,
                                           "result": result,
                                           "message": message, "path": save_pth})
    else:
        app_logger.warning(f'some problems {status}')
        return templates.TemplateResponse("error_form.html",
                                          {"request": request,
                                           "result": status,
                                           "name": file.filename})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
