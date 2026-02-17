# Senior-Project
LLM based healthcare system


This is where we can see the website/code. (https://julianc2009.github.io/Senior-Project/)

Built with OpenAI API

# Requirements
Python 3.10+ needed, Open API key, Python Virtual environment (optional but recommended) 

# How to Run

1. Clone the repo
3. Create a `.env` file:
   OPENAI_API_KEY=your_api_key_here
4. Create a virtual environment (optional)
   Open the command prompt and run:
   python -m venv venv
   venv \Scripts\activate
5. Navigate to the Repo folder in the command prompt
6. Install the python dependencies using the requirements.txt file
   pip install -r requirements.txt
7. Start the server
   uvicorn Server:app --host 0.0.0.0 --port 3000 --reload
8. Open:
   http://localhost:3000

# How to Run Gemini
1. Clone server.js
3. run npm install express @google/generative-ai dotenv in command promt
4. run pip install flask python-dotenv google-generativeai (for python)
5. create a .env file and add the key as GEMINI_API_KEY=your_key_here
6. make sure the file format is - Gemini_API(folder): server.js, .env ->/ Public(Folder inside): index.html(same as the open ai index)
7. Run the server.js
8. Open http://localhost:3000
   

## Notes
Live AI responses require OpenAI API billing.
