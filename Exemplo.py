from PIL import Image
import requests
import base64
import json


url = "https://vision.googleapis.com/v1/images:annotate?key=AIzaSyDi6yw0t-UxDE8gU0JR6q09Xg493DpuKi8"


def splitImageCaptcha(source):

    img = Image.open(source)

    width, height = img.size

    count = 0

    for x in range(5):
        size = (width / 5)
        left = size * count
        right = left + size
        coords = (left, 0, right, height)
        crop = img.crop(coords)

        crop.save("sources/crops/crop"+str(count+1)+".png", "PNG")
        count += 1


def googleVisionApi():

    source = "sources/crops/crop3.png"

    with open(source, "rb") as img_file:
        my_base64 = base64.b64encode(img_file.read())

    data = {
        "requests": [
            {
                "image": {
                    "content": my_base64.decode("utf-8")
                },
                "features": [
                    {"type": "OBJECT_LOCALIZATION"},
                    {"type": "LANDMARK_DETECTION"},
                    {"type": "WEB_DETECTION"},
                ],
            }
        ]
    }

    r = requests.post(url=url, data=json.dumps(data))

    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":

    source = "sources/captcha.png"

    splitImageCaptcha(source)

    googleVisionApi()
