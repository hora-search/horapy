"""
a simple knowledge search web api server
knowledge converted from https://github.com/nikitavoloboev/knowledge/
each section of text is mapped to 512-D vector with 1G bytes U-S-E model
 from tensorflow-hub
searched by nearest neighbor index

run> pyyhon knowledge_search.py
test> curl http://localhost:5000/api/q/what%20is%20hora
"""
from flask import Flask

from flask_cors import CORS
from flask import request, jsonify

import tensorflow_hub as hub
import numpy as np
from horapy import HNSWIndex

app = Flask(__name__)
app.debug = True

CORS(app)

knowledge_dict = dict()
NEWLINE = '\n'
dimension = 512
index = HNSWIndex(dimension, "usize")

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)


def convert_one_file(filename):
    """empty new line separated knowledge block in one file
    converted to 512-D embeddings with universal sentence encoder
    """
    buffer = list()
    progress = 0
    for line in open(filename):
        if line.strip() == '':
            if buffer:
                progress += 1
                message = ''.join(buffer)
                # print(buffer)
                message_embeddings = embed([message])
                message_embedding = np.array(message_embeddings).tolist()[0]
                # print(message_embedding)
                message_embedding_snippet = ", ".join((str(x) for x in message_embedding))
                knowledge_dict[progress] = message
                index.add(np.float32(message_embedding), progress)
                buffer =  list()
                if progress % 10000 == 1:
                    print(progress)
        else:
            buffer.append(line)

convert_one_file('knowledge.txt')
print('build index')
index.build("euclidean")  # build index
print('done')
            

@app.route('/api/q/<query>',  methods=('GET',))
def api_query(query):
    #query = request.args.get('q')
    print(query)
    message_embeddings = embed([query])
    message_embedding = np.array(message_embeddings).tolist()[0]

    results = index.search(message_embedding, 10)
    r = list()
    for i in results:
        r.append(knowledge_dict[i])
    return jsonify(dict(data=r))


if __name__ == '__main__':
    app.run(host="0.0.0.0")
