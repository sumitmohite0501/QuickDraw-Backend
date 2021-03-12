from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

allClasses = ['Bird', 'Flower', 'Hand', 'House', 'Mug', 'Pencil', 'Spoon', 'Sun', 'Tree', 'Umbrella']
# allClasses.sort()
ort_session = ort.InferenceSession('savedModel/model.onnx')


def process(path):
  image = Image.fromarray(plt.imread(path)[:, :, 3])
  # image = Image.open(path)
  # plt.imsave('check1.png',image)
  # arr = np.asarray(image)


  image = image.resize((32, 32))
  image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]

  return image[None]


def test(path):
  image = process(path)

  probs = ort_session.run(None, {'data': image})[0]
  output = probs.argmax()

  # print(allClasses)

  return allClasses[output]

# if __name__ == '__main__':
#   test(path)
