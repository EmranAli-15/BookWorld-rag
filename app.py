import os
from dotenv import load_dotenv
from flask import Flask, request
from flask import jsonify
import pymongo
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
db_url = os.getenv("DB_URL")

app = Flask(__name__)


model = SentenceTransformer('intfloat/multilingual-e5-small')

mongo_client = pymongo.MongoClient(db_url)
db = mongo_client["BookWorld"]
collection = db["vector-data"]



client = genai.Client(api_key=gemini_api_key)


def gemini_response(query, results_list):
    chat = client.chats.create(model="gemini-3-flash-preview")

    context = "\n---\n".join(results_list)
    prompt = f"""
    Context information is below:
    {context}
    
    Given the context information and not prior knowledge, answer the query.
    If the answer is not in the context, politely state that you don't know.
    
    Query: {query}
    Answer:
    """

    response = chat.send_message(prompt)
    print(response.text)
    return response.text


@app.route("/text", methods=["POST"])
def text():
    data = request.get_json()
    print(data)
    return data["name"]



@app.route("/hybrid/<ask>")
def hybrid(ask):
    query_text = ask
    query_vector = model.encode(f"{ask}").tolist()

    vector_results = list(collection.aggregate([
        {
            "$vectorSearch":{
                "index": "vector_index",
                "path":"embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": 5
            }
        },
        {
            "$project": {
                "text": 1,
                "_id": 1
            }
        }
    ]))

    text_results = list(collection.aggregate([
        {
            "$search":{
                "index": "default",
                "text": {
                    "query": query_text,
                    "path": "text"
                }
            }
        },
        {
            "$limit": 5
        },
        {
            "$project": {
                "text": 1,
                "_id": 1
            }
        }
    ]))


    rrf_map = {}
    k = 60

    # Process Vector Results
    for rank, doc in enumerate(vector_results, 1):
        doc_id = str(doc['_id'])
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"score": 0, "text": doc['text']}
    
        rrf_map[doc_id]["score"] += (1 / (k + rank))


    # Process Text Results
    for rank, doc in enumerate(text_results, 1):
        doc_id = str(doc['_id'])
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"score": 0, "text": doc['text']}
    
        rrf_map[doc_id]["score"] += (1 / (k + rank))


    # Convert dictionary to a list and sort by score descending
    sorted_results = sorted(rrf_map.values(), key=lambda x: x['score'], reverse=True)

    top_5_text = [item['text'] for item in sorted_results[:5]]

    get_beautify = gemini_response(query_text, top_5_text)

    return jsonify(get_beautify)





@app.route("/")
def index():
    return "App is running."

if __name__ == "__main__":
    app.run(debug=True)