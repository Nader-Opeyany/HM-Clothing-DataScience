import hashlib
import json
from unittest import TestCase


class IntegrationTestCase(TestCase):
    def hash(self, data):
        return hashlib.md5(json.dumps(data, sort_keys=True).encode("utf8")).hexdigest()
