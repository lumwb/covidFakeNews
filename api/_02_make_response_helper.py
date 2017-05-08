import json
from flask import Flask, make_response

app = Flask(__name__)

books = [{
    'id': 33,
    'title': 'The Raven',
    'author_id': 1
}]


@app.route('/book')
def book_list():
    content = json.dumps(books)

    response = make_response(
        content, 200, {'Content-Type': 'application/json'})
    # Check utils.json_response ;)

    return response
