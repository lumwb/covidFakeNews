from flask import Flask, abort, request
from flask_restful import Resource, Api


app = Flask(__name__)
api = Api(app)

LAST_ID = 33
books = [{
    'id': 33,
    'title': 'The Raven',
    'author_id': 1
}]


# class BookResource(Resource):
#     def get(self, book_id):
#         print("calling book resource get")
#         print("Hello World")
#         book = search_book(books, book_id)
#         if book is None:
#             abort(404)
#         return book

#     def delete(self, book_id):
#         for idx, book in enumerate(books):
#             if book['id'] == book_id:
#                 del books[idx]
#                 return '', 204
#         abort(404)


class CovidFakaeNews(Resource):
    def get(self):
        return "the news is fake / real", 200

    def post(self):
        # global LAST_ID
        # LAST_ID += 1
        # data = request.json
        # book = {
        #     'id': LAST_ID,
        #     'title': data['title'],
        #     'author_id': data['author_id']
        # }
        # books.append(book)
        return "the news is fake / real", 200


api.add_resource(CovidFakaeNews, '/checkFakeNews')
# api.add_resource(BookResource, '/book/<int:book_id>')


@app.errorhandler(404)
def not_found(e):
    return '', 404
