from app.routes import app
from config import *

def worker():
    HOST = apiConfig.get('host')
    PORT = int(apiConfig.get('port'))
    DEBUG = bool(apiConfig.get('debug'))

    from waitress import serve
    serve(app, host=HOST, port=PORT)
    #app.run(host=HOST,port=PORT , debug =DEBUG)

    # main_app = socketio.Middleware(sio, app)
    # eventlet.wsgi.server(eventlet.listen((HOST, PORT)), main_app)

if __name__=='__main__':
    worker()