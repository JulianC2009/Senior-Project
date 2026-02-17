import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder="public", static_url_path="")

# Check API Key
api_key = os.getenv("GEMINI_API_KEY")
print("Checking API Key:", "Loaded" if api_key else "Not Found")

# Configure Gemini
genai.configure(api_key=api_key or "YOUR_BACKUP_KEY_HERE")

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction="You are a helpful Health Assistant. Be concise and professional."
)

# Serve static files 
@app.route("/")
def serve_index():
    return send_from_directory("public", "index.html")


# Chat endpoint
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message")

        print("User said:", message)

        response = model.generate_content(message)
        text = response.text

        print("Gemini replied successfully!")

        return jsonify({"reply": text})

    except Exception as error:
        print("--- GEMINI ERROR ---")
        print(str(error))
        print("--------------------")

        return jsonify({"error": "Gemini API Error: " + str(error)}), 500


# Run server
if __name__ == "__main__":
    app.run(port=3000, debug=True)
