"""Check real app route validation without requiring trained Assignment 4 weights."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


class Assignment4APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_routes_and_validation(self):
        for route in ['/generate_diffusion','/generate_energy']:
            with patch('app.main.image_grid_png', return_value=b'test-png'):
                response = self.client.post(route,json={'count':4,'steps':50,'seed':42})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.headers['content-type'],'image/png')
                for body in [{'count':0},{'count':9},{'steps':0},{'steps':201},{'seed':-1}]:
                    self.assertEqual(self.client.post(route,json=body).status_code,422)

    def test_missing_checkpoint(self):
        with patch('app.main.image_grid_png',side_effect=FileNotFoundError('Train first')):
            for route in ['/generate_diffusion','/generate_energy']:
                self.assertEqual(self.client.post(route,json={}).status_code,503)
