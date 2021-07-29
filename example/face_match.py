# -*- coding: utf-8 -*-
from numpy import imag
from horapy import HNSWIndex
import face_recognition
from os import listdir
from os.path import isfile, join
from flask import Flask, request, abort, jsonify

image2vector = dict()        # image to vector map
f = 128                      # vector dimension
index = HNSWIndex(f, "str")  # init Hora index instance
app = Flask(__name__)        # wep app instance


def img2vector():
    # encoding all image into vector
    face_image_path = ""  # here write you image file path

    images = [f for f in listdir(face_image_path)
              if isfile(join(face_image_path, f))]

    for f in images:
        image = face_recognition.load_image_file(face_image_path + "/" + f)
        encoding = face_recognition.face_encodings(image)
        if encoding and len(encoding) > 0:
            image2vector[f] = encoding
            index.add(encoding, f)

    index.build("euclidean")


@app.route("/match", methods=['GET'])
def image_match():
    # image match main function
    image = request.args.get('image', '')  # get request image query
    if image not in image2vector:          # if thers is not image in the map, return 404
        abort(404)
    response = list()
    for similar_image_idx in index.search(image, 10):
        response.append({
            "image": similar_image_idx,
            "image_encoding": image2vector(similar_image_idx)
        })
    return jsonify(response)


if __name__ == "__main__":
    image2vector()
    app.run(debug=True)
