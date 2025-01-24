import json
from flask import Response, jsonify, render_template, request, stream_with_context

from .app import app
#from .chat_api import call_chat
from .comparator import call_chat

demo_name = "Research Paper Comparator"

@app.route("/")
def index():
    return render_template("index2.html", demo_name=demo_name)
 
@app.route("/chat", methods=['POST'])
def chat_handler():
    # request_message = request.json["message"]
    left = request.json["left"]
    right = request.json["right"]
    print(request)

    @stream_with_context
    def response_stream():
        for chunk in call_chat(left, right):
            # returning a json format for easier encoding
            # each chunk {"token": "..."}
            yield json.dumps(chunk, ensure_ascii=False) + "\n"

    return Response(response_stream(), mimetype="text/event-stream")

@app.route("/user/<user_id>", methods=["GET"])
def get_user(user_id):
    # Retrieve user data from the database and return a JSON response
    return {"name": "Example Name"}

