import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import base64
import os

# export_file_url = "os.environ.get("FILE_URL")"
export_file_url = "https://www.googleapis.com/drive/v3/files/1YFhdexgOfAZtiCVn4eqauy1TQUvXmtP_?alt=media&key=AIzaSyCtFRvO-H1NFGcLJxNhG9bpucP6UqOLd-U"

print(export_file_url)

export_file_name = 'trained_model_98.pkl'
classes = ['edible', 'poisonous']


path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):

    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


def model_predict(img):
    img = open_image(BytesIO(img))
    pred_class, pred_idx, outputs = learn.predict(img)
    print(pred_class)
    formatted_outputs = ["{:.1f}".format(value) for value in [
        x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
        zip(learn.data.classes, map(str, formatted_outputs)),
        key=lambda p: p[1],
        reverse=True
    )
    pred_dict = {i[0]: i[1] for i in pred_probs}
    # for k, v in pred_dict.items():
    #    print(k, v)

    message = {
        'status': 200,
        'message': 'OK',
        'predictions': pred_dict,
    }
    return JSONResponse(message)


def decode(img_b64):
    img = base64.b64decode(img_b64)
    return img


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


@app.route('/predict', methods=["POST"])
async def upload(request):
    if request.method == 'POST':
        # Get the file from post request
        form = await request.form()
        img_bytes = form["image"]
        if img_bytes != None:
            # Make prediction
            img = decode(img_bytes)
            preds = model_predict(img)
            return preds


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host="0.0.0.0",
                    port=int(os.environ.get("PORT", 5000)), log_level="info")
