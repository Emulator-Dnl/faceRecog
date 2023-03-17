"""
Rayas me da el nombre de la imagen (con extensión) del nuevo voluntario
Yo transformo esa imagen a vector y la comparo con los demás vectores de la BD
le devuelvo el nombre de la imagen de quién es muy parecido a ése o nulo

Al medir distancias, me salto a la propia imagen de la que Rayas me pide que
busque similitudes
"""

import faceRecog
fr = faceRecog

from flask import Flask, jsonify
from flask_sslify import SSLify

app = Flask(__name__)
sslify = SSLify(app)

@app.route('/find/<string:filename>')
def find(filename):
    return jsonify({"Response": fr.findSimilar(filename)})

@app.route('/resetDatabase')
def resetDatabase():
    return jsonify({"Response": fr.resetDatabase()})

@app.route('/add/<string:filename>')
def add(filename):
    return jsonify({"Response": fr.add(filename)})

@app.route('/addAll')
def addAll():
    return jsonify({"Response": fr.addAll()})

@app.route('/setFolder/<string:folder>')
def setFolder(folder):
    return jsonify({"Response": fr.setFolder(folder)})

@app.route('/getFolder')
def getFolder():
    return jsonify({"Response": fr.getFolder()})

if __name__ == '__main__':
    app.run(debug=True, port=443)

