import os
from flask import Flask, request
import json
from flask_cors import CORS
import base64
import onnx_save
from datetime import datetime as dt
app = Flask(__name__)
cors = CORS(app)
datasetPath = 'data1'


def findname():
  now = dt.now()
  return now.strftime('%d%m%Y_%H%M%S')


@app.route('/upload_canvas', methods=['POST'])
def upload_canvas():
  print("hello")
  data = json.loads(request.data.decode('utf-8'))
  image_data = data['image'].split(',')[1].encode('utf-8')
  fileName = data['filename']
  className = data['classname']

  os.makedirs(f'{datasetPath}/{className}/image', exist_ok=True)
  with open(f'{datasetPath}/{className}/image/{fileName}', 'wb') as fh:
    fh.write(base64.decodebytes(image_data))

  return "got the image"


@app.route('/Quickdraw/test', methods=['POST'])
def result():
  data = json.loads(request.data.decode('utf-8'))
  image = data['image'].split(',')[1].encode('utf-8')

  os.makedirs('resultdata', exist_ok=True)
  filename = findname() + '.png'

  with open(f'resultdata/{filename}', 'wb') as f:
    f.write(base64.decodebytes(image))

  output = onnx_save.test(f'resultdata/{filename}')
  #print(output)

  return output
