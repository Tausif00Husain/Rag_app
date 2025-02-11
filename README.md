# Rag_app

# Ollama setup:-
-> From this website download ollama setup https://ollama.com/
-> install the app after the installation you will get an ollama terminal where you need to run:-
    ollama run llama3.2
    # this will install llama3.2 model in your system

1) clone the repo:-
git clone "ssh-link"
cd Rag_app

2) create virtual env:-
python -m venv ".venv"

3) activate the virtual env:-
.\.venv\Scripts\activate

4) install all the required packages:-
python install -r requirements.txt

5) To run the server:-
python server.py

6) click on the url shown in the terminal and the endpoint is "/query".

7) for postman:-
 Method-> POST 
 URL-> http://127.0.0.1:5000/query
 Headers-> "Content-Type: application/json"
 Body(raw)-> {"query": "What is the buisness problem"}