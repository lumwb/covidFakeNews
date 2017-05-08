from flask import make_response

JSON_MIME_TYPE = 'application/json'


def search_book(books, book_id):
    for book in books:
        if book['id'] == book_id:
            return book


def json_response(data='', status=200, headers=None):
    headers = headers or {}
    if 'Content-Type' not in headers:
        headers['Content-Type'] = JSON_MIME_TYPE

    return make_response(data, status, headers)
