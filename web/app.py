import json
import os
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(STATIC_DIR, '..', 'models', 'dropout3.npz')
_data = np.load(MODEL_PATH)
W1, b1, W2, b2 = _data['arr_0'], _data['arr_1'], _data['arr_2'], _data['arr_3']
STATIC_FILES = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
}


def predict(pixels):
    x = np.asarray(pixels, dtype=np.float64).reshape(-1)
    h = np.tanh(x @ W1 + b1)
    z = h @ W2 + b2
    z = z - z.max()
    e = np.exp(z)
    probs = e / e.sum()
    return int(np.argmax(probs)), probs.tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        entry = STATIC_FILES.get(self.path)
        if entry is None:
            self.send_error(404)
            return
        filename, content_type = entry
        with open(os.path.join(STATIC_DIR, filename), 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/predict':
            self.send_error(404)
            return
        n = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(n) or b'{}')
        pixels = payload.get('pixels', [])
        if len(pixels) != 784:
            self.send_error(400, 'expected 784 pixels')
            return
        digit, probs = predict(pixels)
        body = json.dumps({'digit': digit, 'probs': probs}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    port = 8000
    print(f'serving on http://localhost:{port}')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
