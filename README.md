# Senior-Project
LLM based healthcare system


This is where we can see the website/code. (https://julianc2009.github.io/Senior-Project/)

Built with OpenAI API

# Requirements
- Python 3.10+
- OpenAI API Key
- MySQL Server
- Python Virtual environment (optional but recommended)

# Database Setup (MySQL)

Create a MySQL database:

Run the database file in MySQL from the databases folder, or open it and simply run the statements.

*Important* Inside the Server.py file, go to the section comment:
MySQL connection helper
Replace these INSERT YOUR statements with your proper database name, MySQL password, and MySQL username to get a proper database connection.

Demo accounts' login information is in the database file and the demologins.txt file.


# How to Run

1. Clone the repo
2. Create a `.env` file:
   OPENAI_API_KEY=your_api_key_here
3. Create a virtual environment (optional)
4. Open the command prompt and run
   python -m venv venv
5. Next
   venv \Scripts\activate
6. Navigate to the Repo folder in the command prompt
7. Install the python dependencies using the requirements.txt file
   pip install -r requirements.txt
   pip install faiss-cpu (for rag pipeline)
   pip install tiktoken
8. run build_text_cache.py
9. run the build_index.py
10. Start the server
   uvicorn Server:app --host 0.0.0.0 --port 3000 --reload
11. Open:
   http://localhost:3000
12. To test the pipeline ask What are common symptoms of diabetes? and it should give you the symptoms in the knowledge base. If you ask for something that isn't in the database it would say that it does not have enough information in the medical database. 

# How to Run Gemini
1. Clone server.js
2. run npm install express @google/generative-ai dotenv in command promt
3. run pip install flask python-dotenv google-generativeai (for python)
4. create a .env file and add the key as GEMINI_API_KEY=your_key_here
5. make sure the file format is - Gemini_API(folder): server.js, .env ->/ Public(Folder inside): index.html(same as the open ai index)
6. Run the server.js
7. Open http://localhost:3000
   

## Notes
Live AI responses require OpenAI API billing.
