from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pickle


from utils.load_model import load
from utils.load_tf_model import load_tf_model
from utils.preprocess_image import preprocess_image
from utils.predict_caption import predict_caption
model = load()
max_length = 492
fe = load_tf_model()
tokenizer = pickle.load(open("static/tokenizer.pkl", "rb"))
print("Loaded All")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
route = "/api/dl/v1"
@app.get(route)
def index():
    return {"message": "Hello World"}


@app.post(f"{route}/predict")
def upload(file: UploadFile = File(...)):
    try:
        print("Loading")
        image = file.file.read()
        feature = preprocess_image(image, fe)
        caption = predict_caption(model, feature, tokenizer, max_length)
        # print("Caption is : ", caption)
        return {"status": True, "caption": caption.strip()}
    except Exception:
        return {"status": False,"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        
