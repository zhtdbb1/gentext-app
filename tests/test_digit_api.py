"""Integration tests; require the existing API models and trained MNIST weights."""
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from app.main import app


class DigitAPITests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_png_response(self):
        response = self.client.post('/generate_digits', json={'count': 16, 'seed': 42})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'image/png')
        image = Image.open(BytesIO(response.content))
        image.load()
        self.assertEqual(image.size, (242, 62))
        self.assertEqual(response.content, self.client.post(
            '/generate_digits', json={'count': 16, 'seed': 42}).content)

    def test_invalid_parameters(self):
        for body in [{'count': 0}, {'count': 65}, {'seed': -1}, {'seed': 2147483648}]:
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/generate_digits', json=body).status_code, 422)

    def test_missing_checkpoint(self):
        with patch('app.main.generate_digit_png', side_effect=FileNotFoundError('Train first')):
            response = self.client.post('/generate_digits', json={})
            self.assertEqual(response.status_code, 503)

    def test_existing_routes(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        for route, body in [('/generate', {'start_word': 'the', 'length': 5}),
                            ('/embedding', {'word': 'hello'}),
                            ('/generate_with_rnn', {'start_word': 'the', 'length': 5})]:
            with self.subTest(route=route):
                self.assertEqual(self.client.post(route, json=body).status_code, 200)
        output = BytesIO()
        Image.new('RGB', (64, 64)).save(output, format='PNG')
        self.assertEqual(self.client.post('/classify', files={
            'file': ('test.png', output.getvalue(), 'image/png')}).status_code, 200)
