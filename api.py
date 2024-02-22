from contextlib import asynccontextmanager
import uvicorn
from unified_normalizer.Normalizer import SkillNormalizerOld, JobNormalizer, LanguageNormalizer, SkillNormalizerNew

from fastapi import FastAPI, HTTPException
import yaml 
import os
import logging

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', config['LOGGING_LEVEL'])
PORT = os.getenv('PORT', config['PORT'])
TRESHOLD_DIRECT_ACCEPT = os.getenv('TRESHOLD_DIRECT_ACCEPT', config['TRESHOLD_DIRECT_ACCEPT'])
TRESHOLD_CANDIDATE_ACCEPT = os.getenv('TRESHOLD_CANDIDATE_ACCEPT', config['TRESHOLD_CANDIDATE_ACCEPT'])

logging.basicConfig(filename='normalizer.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=getattr(logging, LOGGING_LEVEL))

ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    # Load the ML model
    skill_normalizer = SkillNormalizerOld()
    job_normalizer = JobNormalizer()
    language_normalizer = LanguageNormalizer()
    skill_normalizer_new = SkillNormalizerNew()

    ml_models["skill"] = skill_normalizer
    ml_models["job"] = job_normalizer
    ml_models["language"] = language_normalizer
    ml_models["skill_new"] = skill_normalizer_new
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=ml_lifespan_manager)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/normalize/{normalization_type}/")
def normalization(normalization_type: str,
                  name: str,
                  threshold_verified_accept: float = TRESHOLD_DIRECT_ACCEPT,
                  threshold_verified_reject: float = TRESHOLD_CANDIDATE_ACCEPT):
    """
    This endpoint normalizes the input string and returns the normalized string
     * path_param normalization_type: The type of normalization to be used.
     * query_param name: The string to be normalized
     * query_param threshold_verified_accept: The threshold to accept the normalized string as verified
     * query_param threshold_verified_reject: The threshold to reject the normalized string as candidate
     ---
     * return: The normalized string, some additional information on the normalized string and some information on the normalization process
    """
    if normalization_type not in ml_models:
        raise HTTPException(status_code=404,
                            detail=f"Invalid normalization type, the available types are: {', '.join(ml_models.keys())}")
    normalizer = ml_models[f"{normalization_type}"]
    norm, add_info, notes, norm_info = normalizer.normalize(name, threshold_verified_accept, threshold_verified_reject)
    return {"normalized_string": norm, "additional_info": add_info, "notes": notes, "normalization_info": norm_info}

@app.get("/check_normalized/{normalization_type}/")
async def check_normalized(normalization_type: str, norm_string: str, accepted: bool = False):
    """
    This endpoint checks if a normalized string is present in the previously verified or candidate normalizations.
     * path_param normalization_type: The type of normalization to be used.
     * query_param norm_string: The normalized string to be checked
     * query_param accepted: A boolean value to check if the normalized string must be searched in the verified collection or in the candidates collection
    ---
     * return: A JSON with this format:
        {
          "exists": boolean value indicating if the normalized string is present in the collection, 
          "result": the result of the search
          }
    """
    if normalization_type not in ml_models:
        raise HTTPException(status_code=404,
                            detail=f"Invalid normalization type, the available types are: {', '.join(ml_models.keys())}")
    if accepted:
        collection_name = "verified"
    else:
        collection_name = "candidates"

    normalizer = ml_models[f"{normalization_type}"]
    exists, result = normalizer.check_if_normalized_is_in_collection(norm_string, collection_name=collection_name)
    return {"exists": exists, "result": result}

@app.get("/vector_search/{normalization_type}/")
async def vector_search(normalization_type: str,
                        string_to_search: str,
                        limit: int = 1,
                        accepted: bool = False,
                        search_normalized: bool = False):
    """
    This endpoint performs a vector search on a the space of previously normalized strings.
     * path_param normalization_type: The type of normalization to be used.
     * query_param string_to_search: The string to be searched in the collection
     * query_param limit: The maximum number of results to return
     * query_param accepted: A boolean value to choose if the search must be in the verified collection or in the candidates collection
     * query_param search_normalized: A boolean value used to choose if the search must be performed on the normalized strings or on both the raw strings and the normalized strings
    ---
     * return: A JSON with this format:
        {
          "results": list of results from the search, 
          "distances": list of distances of the results from the search string
          }
    """
    if normalization_type not in ml_models:
        raise HTTPException(status_code=404,
                            detail=f"Invalid normalization type, the available types are: {', '.join(ml_models.keys())}")
    if accepted:
        collection_name = "verified"
    else:
        collection_name = "candidates"
    normalizer = ml_models[f"{normalization_type}"]
    results, distances = normalizer.vector_search_on_collection(string_to_search=string_to_search,
                                                                collection_name=collection_name,
                                                                limit=limit,
                                                                search_normalized=search_normalized)
    
    return {"results": results, "distances": distances}

@app.get("/get_embedding/{normalization_type}/")
async def get_embedding(normalization_type: str, word: str):
    """
    This endpoint retrieves the embedding of a word.
     * path_param normalization_type: The type of normalization to be used.
     * query_param word: The word to get the embedding of.
    ---
     * return: A JSON with this format:
        {
          "embedding": list of floats representing the word embedding
        }
    """
    if normalization_type not in ml_models:
        raise HTTPException(status_code=404, 
                            detail=f"Invalid normalization type, the available types are: {', '.join(ml_models.keys())}")
    normalizer = ml_models[f"{normalization_type}"]
    embedding = normalizer.get_word_embedding(word)
    return {"embedding": embedding.tolist()}


@app.get("/cosine_similarity/{normalization_type}/")
async def cosine_similarity(normalization_type: str, input1: str, input2: str):
    """
    This endpoint calculates the cosine similarity between two strings.
        * path_param normalization_type: The type of normalization to be used.
        * query_param input1: The first string to be compared
        * query_param input2: The second string to be compared
    ---
        * return: A JSON with this format:
            {
              "similarity": float value representing the cosine similarity between the two strings
            }
    """

    
    if normalization_type not in ml_models:
        raise HTTPException(status_code=404, 
                            detail=f"Invalid normalization type, the available types are: {', '.join(ml_models.keys())}")
    normalizer = ml_models[f"{normalization_type}"]
    similarity = float(normalizer.cosine_similarity(input1, input2))
    return {"similarity": similarity}



if __name__ == "__main__":
    logging.info("Starting the normalizer service")

    logging.info("Adding skills to the index")
    normalizer = SkillNormalizerOld()
    normalizer.massive_add_to_index()

    logging.info("Adding jobs to the index")
    normalizer = JobNormalizer()
    normalizer.massive_add_to_index()

    logging.info("Adding languages to the index")
    normalizer = LanguageNormalizer()
    normalizer.massive_add_to_index()

    logging.info("Adding New Skill normalizer to the index")
    normalizer = SkillNormalizerNew()
    normalizer.massive_add_to_index()

    
    uvicorn.run("api:app", host="0.0.0.0", port=int(config['PORT']))

