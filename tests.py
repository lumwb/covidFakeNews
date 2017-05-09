import os
import json
import sqlite3
import unittest

from api._01_manual_response_class import app as app_step_1
from api._02_make_response_helper import app as app_step_2
from api._03_post_method import app as app_step_3
from api._04_delete_method import app as app_step_4


class Step1TestCase(unittest.TestCase):
    def setUp(self):
        self.app = app_step_1.test_client()
        self.book_id = 33

    def test_book_list(self):
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')

        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0], {
            'id': 33,
            'title': 'The Raven',
            'author_id': 1
        })

    def test_book_detail_200(self):
        resp = self.app.get('/book/{}'.format(self.book_id))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')

        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(content, {
            'id': 33,
            'title': 'The Raven',
            'author_id': 1
        })

    def test_book_detail_404(self):
        resp = self.app.get('/book/1111')
        self.assertEqual(resp.status_code, 404)


class Step2TestCase(unittest.TestCase):
    def setUp(self):
        self.app = app_step_2.test_client()
        self.book_id = 33

    def test_book_list(self):
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')

        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0], {
            'id': 33,
            'title': 'The Raven',
            'author_id': 1
        })


CREATE_BOOK_TABLE_QUERY = """
create table book (
  id integer primary key autoincrement,
  author_id integer,
  title text not null
);
"""
TESTING_DATABASE_NAME = 'test_library.db'


class BaseDatabaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_step_3.config.update({
            'DATABASE_NAME': TESTING_DATABASE_NAME
        })
        cls.db = sqlite3.connect(TESTING_DATABASE_NAME)
        cls.db.execute(CREATE_BOOK_TABLE_QUERY)
        cls.db.commit()

    @classmethod
    def tearDownClass(cls):
        os.remove(TESTING_DATABASE_NAME)

    def setUp(self):
        self.app = self.APP.test_client()
        self.db.execute("DELETE FROM book;")
        self.db.commit()


class Step3TestCase(BaseDatabaseTestCase):
    APP = app_step_3

    def test_book_creation_correct_parameters(self):
        # Preconditions
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')
        # Precondition: no books in the DB
        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 0)

        # Test
        post_data = {
            'title': 'Ulysses',
            'author_id': 2
        }
        resp = self.app.post('/book',
                             data=json.dumps(post_data),
                             content_type='application/json')

        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.content_type, 'application/json')

        # Postconditions
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')
        # Postcondtion: 2 books
        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 1)

        # Postcondtion: Books correct ID incremented
        ulysses = content[0]

        self.assertEqual(ulysses, {
            'id': 1,
            'title': 'Ulysses',
            'author_id': 2
        })

    def test_book_creation_incorrect_parameters(self):
        # Forget the title argument
        post_data = {
            'author_id': 2
        }
        resp = self.app.post('/book',
                             data=json.dumps(post_data),
                             content_type='application/json')

        self.assertEqual(resp.status_code, 400)
        self.assertTrue('title' in resp.get_data(as_text=True))

    def test_book_creation_incorrect_content_type(self):
        # Forget the title argument
        post_data = {
            'author_id': 2,
            'title': 'Ulysses'
        }
        resp = self.app.post('/book',
                             data=json.dumps(post_data))

        self.assertEqual(resp.status_code, 400)
        self.assertTrue('Content Type' in resp.get_data(as_text=True))


class Step4TestCase(BaseDatabaseTestCase):
    APP = app_step_4

    def setUp(self):
        super(Step4TestCase, self).setUp()
        self.db.execute(("INSERT INTO book (id, author_id, title) "
                         "VALUES (1, 2, 'Ulysses');"))
        self.db.execute(("INSERT INTO book (id, author_id, title) "
                         "VALUES (2, 1, 'The Raven');"))
        self.db.commit()

    def test_delete_books_exists(self):
        # Preconditions
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')

        # Precondition: 2 books pre-loaded in the DB
        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 2)

        # Delete 'The Raven'
        resp = self.app.delete('/book/2')
        self.assertEqual(resp.status_code, 204)

        # Postconditions
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')
        # Postcondtion: 1 books
        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 1)

        # Postcondtion: Only book left is Ulysses
        ulysses = content[0]

        self.assertEqual(ulysses, {
            'id': 1,
            'title': 'Ulysses',
            'author_id': 2
        })

    def test_delete_book_doesnt_exist(self):
        # Preconditions
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')

        # Precondition: 2 books pre-loaded in the DB
        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 2)

        # Delete 'The Raven'
        resp = self.app.delete('/book/20')
        self.assertEqual(resp.status_code, 404)

        # Postconditions
        resp = self.app.get('/book')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, 'application/json')
        # Postcondtion: Same number of books
        content = json.loads(resp.get_data(as_text=True))
        self.assertEqual(len(content), 2)
