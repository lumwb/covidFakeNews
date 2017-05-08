from flask import g, abort

from .utils import json_response
from ._03_post_method import app


@app.route('/book/<int:book_id>', methods=['DELETE'])
def book_delete(book_id):
    params = {'id': book_id}
    query = 'SELECT count(*) FROM book WHERE book.id = :id'
    cursor = g.db.execute(query, params)

    # Check if book exists
    if cursor.fetchone()[0] == 0:
        # Doesn't exist. Return 404.
        abort(404)

    # Delete it
    delete_query = 'DELETE FROM book WHERE book.id = :id'
    g.db.execute(delete_query, {'id': book_id})
    g.db.commit()

    return json_response(status=204)


@app.errorhandler(404)
def not_found(e):
    return '', 404
