from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def chatbot_response():
    user_input = request.args.get('msg')
    return jsonify({"response": get_response(user_input)})

def get_response(message):
    message = message.lower()
    if "open" in message or "timing" in message:
        return "The library is open from 8 AM to 8 PM, Monday through Saturday."
    elif "membership" in message:
        return "Library membership is free for students and staff. You can register online or at the front desk."
    elif "book" in message:
        return "You can search and reserve books through the online catalog on our website."
    elif "return" in message or "due" in message:
        return "Books must be returned within 14 days. Late returns will incur a fine of Rs. 5/day."
    elif "wifi" in message or "internet" in message:
        return "Yes, free Wi-Fi is available for all registered users inside the library."
    else:
        return "Sorry, I didn't understand that. Try asking about timings, membership, or book services."

if __name__ == "__main__":
    app.run(debug=True)
