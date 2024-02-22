import pytest  # noqa: F401
import uvicorn  
import yaml
import requests

from pymilvus import Collection
from pymilvus import connections


################################################
# there tests must be run with the API running #
# the API can be run from the api.py file      #
################################################

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

port = int(config['PORT'])
ALIAS = config['ALIAS']
URI = config['URI']

connections.connect(
  alias=ALIAS,
  uri=URI,
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
    response = requests.get(f"http://localhost:{port}/normalize/skill?name=piiiiiiithon")
    assert response.status_code == 200
    assert response.json()["normalized_string"] == "Python"
    assert response.json()["additional_info"]["type_of_skill"] == 'Hard'
    assert response.json()["notes"] is None
    assert response.json()["normalization_info"]["type"] == 'candidate_indirect_match'
    assert response.json()["normalization_info"]["score"] is None

    response = requests.get(f"http://localhost:{port}/normalize/skill?name=piiiiiiithon")
    assert response.status_code == 200
    assert response.json()["normalized_string"] == "Python"
    assert response.json()["additional_info"]["type_of_skill"] == 'Hard'
    assert response.json()["notes"] is None
    assert response.json()["normalization_info"]["type"] == 'candidate_direct_match'
    assert response.json()["normalization_info"]["score"] > config['TRESHOLD_CANDIDATE_ACCEPT']

    expr = "raw_string == 'piiiiiiithon'"
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
    assert response.json()["result"]['raw_string'] == 'Screenwriter'
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
    response = requests.get(f"http://localhost:{port}/vector_search/language?string_to_search=italy&limit=10")
    assert response.status_code == 200
    assert "results" in response.json()
    assert response.json()["results"][-1]['raw_string'] == 'italian'
    assert len(response.json()["distances"]) == 10
    # TBD
    response = requests.get(f"http://localhost:{port}/vector_search/skill_new?string_to_search=italy&search_normalized=True&limit=10")
    assert response.status_code == 200
    assert "results" in response.json()
    assert response.json()["results"][-1]['raw_string'] == response.json()["results"][-1]['norm_string']

    # wrong normalization type
    response = requests.get(f"http://localhost:{port}/vector_search/not_a_norm?string_to_search=italy&limit=10")
    assert response.status_code == 404

def test_api_get_embedding():
    response = requests.get(f"http://localhost:{port}/get_embedding/skill?word=python")
    assert response.status_code == 200
    assert "embedding" in response.json()
    embedding1 = response.json()["embedding"]

    # the embedding should be the same
    response = requests.get(f"http://localhost:{port}/get_embedding/job?word=python")
    assert response.status_code == 200
    assert "embedding" in response.json()
    embedding2 = response.json()["embedding"]
    assert embedding1 == embedding2

    # the embedding should be different
    response = requests.get(f"http://localhost:{port}/get_embedding/skill?word=java")
    assert response.status_code == 200
    assert "embedding" in response.json()
    embedding3 = response.json()["embedding"]
    assert embedding1 != embedding3

    # wrong normalization type
    response = requests.get(f"http://localhost:{port}/get_embedding/not_a_norm?word=java")
    assert response.status_code == 404


def test_api_cosine_similarity():
    response = requests.get(f"http://localhost:{port}/cosine_similarity/skill?input1=python&input2=java")
    assert response.status_code == 200
    assert "similarity" in response.json()
    similarity = response.json()["similarity"]
    assert similarity < 1














