import sys
if sys.version_info <  (3,0):
	from BaseHTTPServer import HTTPServer
	from SimpleHTTPServer import SimpleHTTPRequestHandler
else:
	from http.server import HTTPServer, SimpleHTTPRequestHandler
 
httpd = HTTPServer(('127.0.0.1', 8080), SimpleHTTPRequestHandler)
print('Test server is launched.')
print('Please navigate to http://localhost:8080/rbn_mnist.html')
httpd.serve_forever()
