from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from main import icebreak

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    person_info, profile_pic_url = icebreak(name=name)

    return jsonify({
        "summary": person_info.summary,
        "interests": person_info.topics_of_interest,
        "facts": person_info.facts,
        "ice_breaker": person_info.ice_breakers,
        "picture_url": profile_pic_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
