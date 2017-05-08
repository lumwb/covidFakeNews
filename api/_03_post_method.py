import json
import sqlite3
from flask import Flask, request, g
from .utils import json_response, JSON_MIME_TYPE


app = Flask(__name__)


@app.before_request
def before_request():
    g.db = sqlite3.connect(app.config['DATABASE_NAME'])


@app.route('/book')
def book_list():
    cursor = g.db.execute('SELECT id, author_id, title FROM book;')
    books = [{
        'id': row[0],
        'author_id': row[1],
        'title': row[2]
    } for row in cursor.fetchall()]

    return json_response(json.dumps(books))


@app.route('/book', methods=['POST'])
def book_create():
    if request.content_type != JSON_MIME_TYPE:
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)

    data = request.json
    if not all([data.get('title'), data.get('author_id')]):
        error = json.dumps({'error': 'Missing field/s (title, author_id)'})
        return json_response(error, 400)

    query = ('INSERT INTO book ("author_id", "title") '
             'VALUES (:author_id, :title);')
    params = {
        'title': data['title'],
        'author_id': data['author_id']
    }
    g.db.execute(query, params)
    g.db.commit()

    return json_response(status=201)
