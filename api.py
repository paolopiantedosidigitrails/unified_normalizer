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
     * param normalization_type: The type of normalization to be used, the available types are: skill, job, language
     * param name: The string to be normalized
     * param threshold_verified_accept: The threshold to accept the normalized string as verified
     * param threshold_verified_reject: The threshold to reject the normalized string as candidate
     ---
     * return: The normalized string, some additional information on the normalized string and some information on the normalization process
    """
    if normalization_type not in ml_models:
        raise HTTPException(status_code=404,
                            detail=f"Invalid normalization type, the available types are: {', '.join(ml_models.keys())}")
    normalizer = ml_models[f"{normalization_type}"]
    return normalizer.normalize(name, threshold_verified_accept, threshold_verified_reject)

@app.get("/check_normalized/{normalization_type}/")
async def check_normalized(normalization_type: str, norm_string: str, accepted: bool = False):

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
    if normalization_type not in ml_models:
        raise HTTPException(status_code=404, 
                            detail=f"Invalid normalization type, the available types are: {', '.join(ml_models.keys())}")
    normalizer = ml_models[f"{normalization_type}"]
    embedding = normalizer.get_word_embedding(word)
    return {"embedding": embedding.tolist()}


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

