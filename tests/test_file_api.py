import pytest  # noqa: F401
import yaml
import requests
import os

from pymilvus import Collection
from pymilvus import connections


################################################
# there tests must be run with the API running #
# the API can be run from the api.py file      #
################################################

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

port = int(8855)
ALIAS = config['ALIAS']
MILVUS_HOST = os.getenv('MILVUS_HOST', config['MILVUS_HOST'])
MILVUS_PORT = os.getenv('MILVUS_PORT', config['MILVUS_PORT'])

connections.connect(
  alias=ALIAS,
  uri=f"{MILVUS_HOST}:{MILVUS_PORT}",
#   token=MILVUS_TOKEN,
)

def test_dummy():
    assert 1 == 1

#### API TESTS
def test_api_read_root():
    response = requests.get(f"http://localhost:{port}/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_api_normalize():
    # response = requests.get(f"http://localhost:{port}/normalize/skill?name=piiithon")
    # assert response.status_code == 200
    # assert response.json()["normalized_string"] == "Python"
    # assert response.json()["additional_info"]["type_of_skill"] == 'Hard'
    # assert response.json()["notes"] is None
    # assert response.json()["normalization_info"]["type"] == 'candidate_indirect_match'
    # assert response.json()["normalization_info"]["score"] is None

    response = requests.get(f"http://localhost:{port}/normalize/skill?name=piiiithon")
    assert response.status_code == 200
    assert response.json()["normalized_string"] == "Python"
    assert response.json()["additional_info"]["type_of_skill"] == 'Hard'
    assert response.json()["notes"] is None
    assert response.json()["normalization_info"]["type"] == 'candidate_indirect_match'
    assert response.json()["normalization_info"]["score"] is None

    response = requests.get(f"http://localhost:{port}/normalize/skill?name=piiiithon")
    assert response.status_code == 200
    assert response.json()["normalized_string"] == "Python"
    assert response.json()["additional_info"]["type_of_skill"] == 'Hard'
    assert response.json()["notes"] is None
    assert response.json()["normalization_info"]["type"] == 'candidate_direct_match'
    assert response.json()["normalization_info"]["score"] > config['TRESHOLD_CANDIDATE_ACCEPT']

    expr = "raw_string in ['piiiiiiithon', 'piiithon', 'piiiithon']"
    collection = Collection("skill_idx_candidates")
    collection.delete(expr)

    # wrong normalization type
    response = requests.get(f"http://localhost:{port}/normalize/not_a_norm?name=piiiiiiithon")
    assert response.status_code == 404

def test_api_check_normalized():
    response = requests.get(f"http://localhost:{port}/check_normalized/job?norm_string=Screenwriter&accepted=True")
    assert response.status_code == 200
    assert "exists" in response.json()
    assert response.json()["exists"]
    assert response.json()["result"]['raw_string'] == 'screenwriter'
    assert response.json()["result"]['norm_string'] == 'Screenwriter'
    assert response.json()["result"]['notes'] is None

    response = requests.get(f"http://localhost:{port}/normalize/skill?name=Horse_breading")
    assert response.status_code == 200
    normalized_string = response.json()["normalized_string"]
    response = requests.get(f"http://localhost:{port}/check_normalized/skill?norm_string={normalized_string}")
    assert response.status_code == 200
    assert "exists" in response.json()
    assert response.json()["result"]['norm_string'] == normalized_string
    response = requests.get(f"http://localhost:{port}/check_normalized/skill?norm_string=Horse_breading&accepted=True")
    assert response.status_code == 200
    assert "exists" in response.json()
    assert not response.json()["exists"]

    expr = f"norm_string == '{normalized_string}'"
    collection = Collection("skill_idx_candidates")
    collection.delete(expr)

    # wrong normalization type
    response = requests.get(f"http://localhost:{port}/check_normalized/not_a_norm?norm_string=python")
    assert response.status_code == 404

def test_api_vector_search():
    response = requests.get(f"http://localhost:{port}/vector_search/language?string_to_search=italy&limit=10&accepted=True")
    assert response.status_code == 200
    assert "results" in response.json()
    assert response.json()["results"][-1]['raw_string'] == 'italian '
    assert len(response.json()["distances"]) == 10

    # wrong normalization type
    response = requests.get(f"http://localhost:{port}/vector_search/not_a_norm?string_to_search=italy&limit=10")
    assert response.status_code == 404

def test_api_get_embedding():

    response = requests.post(f"http://localhost:{port}/get_embedding/skill/", json=["string"])
    assert response.status_code == 200
    assert "embeddings" in response.json()
    embedding1 = response.json()["embeddings"][0]

    # the embedding should be the same
    response = requests.post(f"http://localhost:{port}/get_embedding/job/", json=["string"])
    assert response.status_code == 200
    assert "embeddings" in response.json()
    embedding2 = response.json()["embeddings"][0]
    assert embedding1 == embedding2

    # the embedding should be different
    response = requests.post(f"http://localhost:{port}/get_embedding/skill/", json=["test"])
    assert response.status_code == 200
    assert "embeddings" in response.json()
    embedding3 = response.json()["embeddings"][0]
    assert embedding1 != embedding3

    # wrong normalization type
    response = requests.post(f"http://localhost:{port}/get_embedding/not_a_norm/", json=["test"])
    assert response.status_code == 404

    #test multiple words
    response = requests.post(f"http://localhost:{port}/get_embedding/skill/", json=["test", "string"])
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert len(response.json()["embeddings"]) == 2


def test_api_cosine_similarity():
    response = requests.get(f"http://localhost:{port}/cosine_similarity/skill?input1=python&input2=java")
    assert response.status_code == 200
    assert "similarity" in response.json()
    similarity = response.json()["similarity"]
    assert similarity < 1














