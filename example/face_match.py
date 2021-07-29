# -*- coding: utf-8 -*-
import numpy as np
from horapy import HNSWIndex
import face_recognition
from os import listdir
from os.path import isfile, join
from flask import Flask, request, abort, jsonify
from PIL import Image

image2vector = dict()        # image to vector map
f = 128                      # vector dimension
index = HNSWIndex(f, "str")  # init Hora index instance
app = Flask(__name__)        # wep app instance


def img2vector():
    # encode all image into vector
    face_image_path = ""  # here write you image file path

    images = [f for f in listdir(face_image_path)
              if isfile(join(face_image_path, f))]

    for f in images:
        image = face_recognition.load_image_file(face_image_path + "/" + f)
        embedding = face_recognition.face_encodings(image)
        if embedding and len(embedding) > 0:
            image2vector[f] = embedding[0]
            index.add(embedding[0], f)

    index.build("euclidean")


@app.route("/match", methods=['GET'])
def image_match():
    # image match main function
    image = request.args.get('image', '')  # get request image query
    if image not in image2vector:          # if thers is not image in the map, return 404
        abort(404)
    response = list()
    for similar_image_idx in index.search(image2vector[image], 10):
        response.append({
            "image": similar_image_idx,
            "image_embedding": image2vector[similar_image_idx].tolist()
        })
    return jsonify(response)

@app.route("/search", methods=['POST'])
def image_search():
    img = Image.open(request.files['file'].stream)  # get request image query
    image = np.array(img)
    embedding = face_recognition.face_encodings(image) # encode incomming image into vector
    response = list()
    for similar_image_idx in index.search(embedding, 10):
        response.append({
            "image": similar_image_idx,
            "image_embedding": image2vector[similar_image_idx].tolist()
        })
    return jsonify(response)

if __name__ == "__main__":
    img2vector()
    app.run(debug=True)
