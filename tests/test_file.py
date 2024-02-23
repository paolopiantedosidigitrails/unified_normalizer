import pytest  # noqa: F401
from unified_normalizer.Normalizer import Normalizer, SkillNormalizerOld, LanguageNormalizer, JobNormalizer, SkillNormalizerNew
import time 

import numpy as np
from pymilvus import utility

def test_dummy():
    assert 1 == 1

def test_word_embedding():
    normalizer = Normalizer('test')
    assert len(normalizer.get_word_embedding("test")) == 768
    

    # timing test
    t = time.time()
    for x in range(100):
        normalizer.get_word_embedding(f"test+{x}")
    t = time.time() - t
    print(f"100 word embeddings in {t} seconds. single time: {t/100} seconds.")
    assert t < 5

    # fast timing test, from cache
    t = time.time()
    for x in range(100):
        normalizer.get_word_embedding(f"test+{x}")
    t = time.time() - t
    print(f"100 word embeddings in {t} seconds. (from cache) single time: {t/100} seconds.")
    assert t < 1

    # check if embedding is normalized
    vector = normalizer.get_word_embedding("test")
    assert np.linalg.norm(vector) > 0.99

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_cosine_similarity():
    normalizer = Normalizer('test')
    assert normalizer.cosine_similarity("test", "test") == 1.0
    assert normalizer.cosine_similarity("test", "test2") < 1.0
    assert normalizer.cosine_similarity("test", "test2") > 0.0

    vector = normalizer.get_word_embedding("testerino")
    assert len(vector) == 768
    assert normalizer.cosine_similarity(vector, "test") < 1.0
    assert normalizer.cosine_similarity(vector, "test") > 0.0

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_massive_add_to_index_and_norm_finder():
    normalizer = Normalizer('test',
                             header_raw_string="test_skill_raw",
                             header_norm_string="test_skill_normalized",
                             header_additional_info=['test_domain','test_field','test_type_of_skill'])
    utility.has_collection('test_idx_candidates')
    normalizer.massive_add_to_index("./data/test.csv")

    # check if all data is in the index
    resp, value = normalizer.check_if_normalized_is_in_collection('.NET_normed', 'candidates')
    assert resp is True
    assert value['additional_info']['test_domain'] == 'Computer Science test, Software Engineering'

    resp, value = normalizer.check_if_normalized_is_in_collection('.NET_normed', 'verified')
    assert resp is True
    assert value['additional_info']['test_domain'] == 'Computer Science test, Software Engineering'

    resp, value = normalizer.check_if_normalized_is_in_collection('NET_normed', 'candidates')
    assert resp is False
    assert value is None
    
    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")


def test_prompt_element_collection_creation():
    normalizer = Normalizer('test', header_raw_string="raw_string", header_norm_string="norm_string", header_additional_info=["additional_info_key"])
    normalizer.add_to_index("verified", "test_raw", "test_normalized", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "test_rawion", "test_raw", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "test_", "test", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "raw_testing", "test_normalized", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "armadillo", "test_normalized", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "armadillo2", "test_normalized", {"additional_info_key": "additional_info_value"})

    response, distances = normalizer.vector_search_on_collection(string_to_search= 'test',
                                                      collection_name = 'verified',
                                                      limit=20)

    assert response[0]['raw_string'] == 'armadillo'
    assert response[1]['raw_string'] == 'armadillo2'

    assert distances[0] < distances[1]

    t = time.time()
    for x in range(100):
        normalizer.vector_search_on_collection(string_to_search= 'test',
                                                      collection_name = 'verified',
                                                      limit=1)
    t = time.time() - t
    print(f"100 searches in {t} seconds. single time: {t/100} seconds.")

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_prompt_creation_skills():
    normalizer = SkillNormalizerOld()
    normalizer.massive_add_to_index()
    response, distance =  normalizer.vector_search_on_collection(string_to_search= 'test',
                                                                collection_name = 'verified',
                                                                limit=1)
    prompt = normalizer.create_prompt("test", response)
    assert prompt[:54] == "skill_raw;skill_normalized;domain;field;type_of_skill\n"
    assert prompt[-6:] == "\ntest;"

    utility.drop_collection("skill_idx_verified")
    utility.drop_collection("skill_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_LLM_call_skill():
    normalizer = SkillNormalizerOld()
    normalizer.massive_add_to_index()
    response, distance =  normalizer.vector_search_on_collection(string_to_search= 'test', collection_name = 'verified', limit=10)
    norm, additional_info, notes = normalizer.LLM_call('random_skill', response)
    assert notes is None
    assert norm is not None
    assert additional_info is not None
    assert additional_info.get('domain') is not None
    assert additional_info.get('field') is not None
    assert additional_info.get('type_of_skill') is not None

    utility.drop_collection("skill_idx_verified")
    utility.drop_collection("skill_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_LLM_call_with_notes():
    normalizer = Normalizer('test',
                             header_raw_string="raw_string",
                             header_norm_string="norm_string",
                             header_additional_info=["additional_info_key"],
                             header_notes=["test_note"])
    normalizer.add_to_index("verified", "test_raw", "test_normalized", {"additional_info_key": "additional_info_value"}, {"test_note": "test_note_value"})
    response, distance =  normalizer.vector_search_on_collection(string_to_search= 'test_normalized', collection_name = 'verified', limit=10)
    norm, additional_info, notes = normalizer.LLM_call('test_normalized', response) 
    assert norm == "test_normalized"
    assert additional_info['additional_info_key'] == "additional_info_value"
    assert notes['test_note'] == "test_note_value"

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_search_on_normalized():
    normalizer = Normalizer('test')
    normalizer.add_to_index("verified", "test_raw", "test_normalized")
    normalizer.add_to_index("verified", "test_raw2", "test_normalized")
    normalizer.add_to_index("verified", "test_normalized", "test_normalized")
    normalizer.add_to_index("verified", "cane", "cane")
    hits = normalizer.vector_search_on_collection(string_to_search= 'test_normalized',
                                                  collection_name = 'verified',
                                                  limit=20,
                                                  search_normalized=True)
    assert len(hits[0]) == 2
    assert hits[0][1]['raw_string'] == 'test_normalized'
    assert hits[0][1]['norm_string'] == 'test_normalized'
    assert hits[0][1].get('additional_info') is None
    
    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

    del normalizer

    normalizer = Normalizer('test', header_raw_string="raw_string", header_norm_string="norm_string", header_additional_info=["additional_info_key"])
    normalizer.add_to_index("verified", "test_raw", "test_normalized", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "test_raw2", "test_normalized", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "test_normalized", "test_normalized", {"additional_info_key": "additional_info_value"})
    normalizer.add_to_index("verified", "cane", "cane", {"additional_info_key": "additional_info_value"})

    hits = normalizer.vector_search_on_collection(string_to_search= 'test_normalized',
                                                    collection_name = 'verified',
                                                    limit=20,
                                                    search_normalized=True)
    assert len(hits[0]) == 2
    assert hits[0][1]['raw_string'] == 'test_normalized'
    assert hits[0][1]['norm_string'] == 'test_normalized'
    assert hits[0][1]['additional_info']['additional_info_key'] == 'additional_info_value'

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")


def test_search_on_normalized_with_notes():
    normalizer = Normalizer('test',
                             header_raw_string="raw_string",
                             header_norm_string="norm_string",
                             header_additional_info=["additional_info_key"],
                             header_notes=["test_note"])
    normalizer.add_to_index("verified", "test_raw", "test_normalized", {"additional_info_key": "additional_info_value"}, {"test_note": "test_note_value"})
    normalizer.add_to_index("verified", "test_normalized", "test_normalized", {"additional_info_key": "additional_info_value"}, {"test_note": "test_note_value"})
    hits = normalizer.vector_search_on_collection(string_to_search= 'test_normalized',
                                                    collection_name = 'verified',
                                                    limit=20,
                                                    search_normalized=True)
    assert len(hits[0]) == 1
    assert hits[0][0]['raw_string'] == 'test_normalized'
    assert hits[0][0]['norm_string'] == 'test_normalized'
    assert hits[0][0]['additional_info']['additional_info_key'] == 'additional_info_value'
    assert hits[0][0]['notes']['test_note'] == 'test_note_value'

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")


def test_prompt_creation():
    normalizer = Normalizer('test', header_raw_string="raw_string",
                             header_norm_string="norm_string",
                             header_additional_info=["additional_info_key", "add_key_2"])
    normalizer.add_to_index("verified", "test_raw", "test_normalized", {"additional_info_key": "additional_info_value", "add_key_2": "add2"})
    normalizer.add_to_index("verified", "test_rawion", "test_raw", {"additional_info_key": "additional_info_value", "add_key_2": "add2"})
    normalizer.add_to_index("verified", "test_", "test", {"additional_info_key": "additional_info_value", "add_key_2": "add2"})
    normalizer.add_to_index("verified", "raw_testing", "test_normalized", {"additional_info_key": "additional_info_value", "add_key_2": "add2"})
    normalizer.add_to_index("verified", "armadillo", "armadillo", {"additional_info_key": "additional_info_value", "add_key_2": "add2"})
    response, distance =  normalizer.vector_search_on_collection(string_to_search= 'test_at_the_end', collection_name = 'verified', limit=20)
    prompt = normalizer.create_prompt("test_at_the_end", response)
    assert prompt[:53] == "raw_string;norm_string;additional_info_key;add_key_2\n"
    assert prompt[:62] == "raw_string;norm_string;additional_info_key;add_key_2\narmadillo"
    assert prompt[-17:] == "\ntest_at_the_end;"

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_normalize():
    normalizer = Normalizer('test',
                             header_raw_string="raw_string",
                             header_norm_string="norm_string",
                             header_additional_info=["animal_type", "food_flag"])

    normalizer.add_to_index("verified", "Dog", "Dog", {"animal_type": "canine", "food_flag": "no"})
    normalizer.add_to_index("verified", "CCane", "Dog", {"animal_type": "canine", "food_flag": "no"})
    normalizer.add_to_index("verified", "gatto", "Cat", {"animal_type": "feline", "food_flag": "no"})
    normalizer.add_to_index("verified", "catt", "Cat", {"animal_type": "feline", "food_flag": "no"})
    normalizer.add_to_index("verified", "cat", "Cat", {"animal_type": "feline", "food_flag": "no"})
    normalizer.add_to_index("verified", "cow", "Cow", {"animal_type": "bovine", "food_flag": "yes"})
    normalizer.add_to_index("verified", "cattle", "Cow", {"animal_type": "bovine", "food_flag": "yes"})
    normalizer.add_to_index("verified", "sheeep", "Sheep", {"animal_type": "ovine", "food_flag": "yes"})
    normalizer.add_to_index("verified", "goat", "Goat", {"animal_type": "ovine", "food_flag": "yes"})
    normalizer.add_to_index("verified", "goats", "Goat", {"animal_type": "ovine", "food_flag": "yes"})
    normalizer.add_to_index("verified", "rat", "Rat", {"animal_type": "rodent", "food_flag": "no"})
    normalizer.add_to_index("verified", "mouse", "Mouse", {"animal_type": "rodent", "food_flag": "no"}, flush_db=True)

    norm, additional_info, notes, normalization_info = normalizer.normalize("dog")
    assert notes is None
    assert norm == "Dog"
    assert additional_info['animal_type'] == "canine"
    assert additional_info['food_flag'] == "no"
    assert normalization_info['score'] > 0.9
    assert normalization_info['type'] == "verified_direct_match"

    norm, additional_info, notes, normalization_info = normalizer.normalize("trichecho", threshold_verified_accept=0.0)
    assert notes is None
    assert norm == "Cat"
    assert additional_info['animal_type'] == "feline"
    assert additional_info['food_flag'] == "no"
    assert normalization_info['score'] < 0.9
    assert normalization_info['type'] == "verified_direct_match"

    norm, additional_info, notes, normalization_info = normalizer.normalize("trichecho")
    assert notes is None
    assert norm == "Manatee"
    assert additional_info['animal_type'] == "herbivore"
    assert additional_info['food_flag'] == "yes"
    assert normalization_info['score'] is None
    assert normalization_info['type'] == "new_normalized_string"

    norm, additional_info, notes, normalization_info = normalizer.normalize("manetee", threshold_verified_accept=1.1, threshold_candidate_accept=0.0)
    assert notes is None
    assert norm == "Manatee"
    assert additional_info['animal_type'] == "herbivore"
    assert additional_info['food_flag'] == "yes"
    assert normalization_info['score'] is not None
    assert normalization_info['type'] == "candidate_direct_match"

    norm, additional_info, notes, normalization_info = normalizer.normalize("dog & woolf & pitbull")
    assert notes is None
    assert norm == "Dog"
    assert additional_info['animal_type'] == "canine"
    assert additional_info['food_flag'] == "no"
    assert normalization_info['score'] is None
    assert normalization_info['type'] == "candidate_indirect_match"

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")


def test_check_if_normalized_is_in_collection():
    normalizer = Normalizer('test', header_raw_string="raw_string", header_norm_string="norm_string", header_additional_info=["animal_type", "food_flag"])
    normalizer.add_to_index("verified", "Dogggg", "Dog", {"animal_type": "canine", "food_flag": "no"})
    flag, ans = normalizer.check_if_normalized_is_in_collection("Dog", "verified")
    assert flag is True
    assert ans['raw_string'] == "Dogggg"
    assert ans['norm_string'] == "Dog"
    assert ans['additional_info']['animal_type'] == "canine"
    assert ans['additional_info']['food_flag'] == "no"

    flag, ans = normalizer.check_if_normalized_is_in_collection("Dog", "candidates")
    assert flag is True
    assert ans['raw_string'] == "Dogggg"
    assert ans['norm_string'] == "Dog"
    assert ans['additional_info']['animal_type'] == "canine"
    assert ans['additional_info']['food_flag'] == "no"

    flag, ans = normalizer.check_if_normalized_is_in_collection("not dog", "candidates")
    assert flag is False
    assert ans is None

    flag, ans = normalizer.check_if_normalized_is_in_collection("not dog", "verified")
    assert flag is False
    assert ans is None

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    

def test_language_normalizer():
    normalizer = LanguageNormalizer()
    normalizer.massive_add_to_index()

    norm, additional_info, notes, norm_info = normalizer.normalize('Langues de Signes Allemandes')
    assert notes is None
    assert norm == "german sign language"
    assert additional_info['sign_language'] == '1'
    assert norm_info['type'] == "new_normalized_string"

    norm, additional_info, notes, norm_info = normalizer.normalize('LSF')
    assert notes is None
    assert norm == "french sign language"
    assert additional_info['sign_language'] == '1'
    assert norm_info['type'] == "candidate_indirect_match"
  
    assert norm_info['score'] is None
    
    utility.drop_collection("language_idx_verified")
    utility.drop_collection("language_idx_candidates")
    utility.drop_collection("cache_raw_embedding")


def test_data_retention():
    normalizer1 = Normalizer('test',
                             header_raw_string="raw_string",
                             header_norm_string="norm_string",
                             header_additional_info=["animal_type", "food_flag"])
    normalizer1.add_to_index("verified", "dog", "Dog", {"animal_type": "canine", "food_flag": "no"})
    normalizer1.add_to_index("verified", "cat", "Cat", {"animal_type": "feline", "food_flag": "no"})
    normalizer1.normalize("walrus")
    del normalizer1

    # check if normalizer1 exist:
    try:
        normalizer1
        assert False
    except NameError:
        assert True
    
    # check if normalizer2 can still access the data
    normalizer2 = Normalizer('test')
    norm, additional_info, notes, normalization_info = normalizer2.normalize("dog")
    assert notes is None
    assert norm == "Dog"
    assert additional_info['animal_type'] == "canine"
    assert additional_info['food_flag'] == "no"
    assert normalization_info['score'] > 0.9

    norm, additional_info, notes, normalization_info = normalizer2.normalize("walrus")
    assert notes is None
    assert norm == "Walrus"
    assert normalization_info['type'] == "candidate_direct_match"

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_massive_add_skip():

    t = time.time()
    normalizer = LanguageNormalizer()
    normalizer.massive_add_to_index()
    t = time.time() - t

    t2 = time.time()
    normalizer = LanguageNormalizer()
    normalizer.massive_add_to_index()
    t2 = time.time() - t2
    print(f"First massive add took {t} seconds, second took {t2} seconds")

    assert t2 < t

    utility.drop_collection("language_idx_verified")
    utility.drop_collection("language_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_notes():
    normalizer = Normalizer('test',
                            header_raw_string="raw_string",
                            header_norm_string="norm_string",
                            header_additional_info=["animal_type", "food_flag"],
                            header_notes=["note"])
    normalizer.add_to_index("verified", "Dog", "Dog", {"animal_type": "canine", "food_flag": "no"}, {"note": "we do not eat dogs"})
    normalizer.add_to_index("verified", "CCane", "Dog", {"animal_type": "canine", "food_flag": "no"}, {"note": "Cane is italian for dogs and we do not eat dogs"})
    normalizer.add_to_index("verified", "gatto", "Cat", {"animal_type": "feline", "food_flag": "no"}, {'note': 'gatto is italian for cat and we do not eat cats'})
    normalizer.add_to_index("verified", "catt", "Cat", {"animal_type": "feline", "food_flag": "no"}, {'note': 'catt is a typo for cat and we do not eat cats'})
    normalizer.add_to_index("verified", "cat", "Cat", {"animal_type": "feline", "food_flag": "no"}, {'note': 'we do not eat cats'})
    normalizer.add_to_index("verified", "cow", "Cow", {"animal_type": "bovine", "food_flag": "yes"}, {'note': 'we eat cows'})
    normalizer.add_to_index("verified", "cattle", "Cow", {"animal_type": "bovine", "food_flag": "yes"}, {'note': 'cattle is a group of cows and we eat cows'})
    normalizer.add_to_index("verified", "sheeep", "Sheep", {"animal_type": "ovine", "food_flag": "yes"}, {'note': 'sheeep is a typo for sheep and we eat sheep'})
    normalizer.add_to_index("verified", "goat", "Goat", {"animal_type": "ovine", "food_flag": "yes"}, {'note': 'we eat goats'})
    normalizer.add_to_index("verified", "goats", "Goat", {"animal_type": "ovine", "food_flag": "yes"}, {'note': 'goats is a group of goats and we eat goats'})
    normalizer.add_to_index("verified", "rat", "Rat", {"animal_type": "rodent", "food_flag": "no"}, {'note': 'we do not eat rats'})
    normalizer.add_to_index("verified", "mouse", "Mouse", {"animal_type": "rodent", "food_flag": "no"}, {'note': 'we do not eat mice'}, flush_db=True)
    
    hits = normalizer.vector_search_on_collection(string_to_search= 'sheeep',
                                                    collection_name = 'verified',
                                                    limit=20)
    assert hits[0][-1]['notes']['note'] == 'sheeep is a typo for sheep and we eat sheep'
    prompt = normalizer.create_prompt("sheeep", hits[0])

    assert prompt[:50] == 'raw_string;note;norm_string;animal_type;food_flag\n'
    assert prompt[-8:] == '\nsheeep;'

    norm, additional_info, notes, norm_info = normalizer.normalize("sheep")
    assert notes['note'] is not None
    assert norm == "Sheep"
    assert additional_info['animal_type'] == "ovine"
    assert additional_info['food_flag'] == "yes"
    assert norm_info['type'] == "candidate_indirect_match"

    utility.drop_collection("test_idx_verified")
    utility.drop_collection("test_idx_candidates")
    utility.drop_collection("cache_raw_embedding")

def test_job_role_normalizer():
    normalizer = JobNormalizer()
    normalizer.massive_add_to_index()

    norm, additional_info, notes, norm_info = normalizer.normalize('Data Scientist')
    assert notes is None
    assert norm == "Data Scientist"
    assert additional_info['general_cluster'] == 'Data and Analytics'
    assert additional_info['function'] == 'Data Science'
    assert norm_info['type'] == "verified_direct_match"

    utility.drop_collection("job_idx_verified")
    utility.drop_collection("job_idx_candidates")
    utility.drop_collection("cache_raw_embedding")


def test_new_skill_normalizer():
    normalizer = SkillNormalizerNew()

    norm, additional_info, notes, norm_info = normalizer.normalize('Data Sciennce')

    assert notes is not None
    assert notes.get('observations') is not None
    assert norm == "Data Science"
    assert additional_info.get('possible_domains') is not None
    assert additional_info.get('possible_fields') is not None
    assert additional_info.get('too_generic')=='0'
    assert norm_info.get('type') == 'new_normalized_string'
    assert norm_info.get('score') is None

    time.sleep(1)
    norm, additional_info, notes, norm_info = normalizer.normalize('Data Sciennce')

    assert notes is not None
    assert notes.get('observations') is not None
    assert norm == "Data Science"
    assert additional_info.get('possible_domains') is not None
    assert additional_info.get('possible_fields') is not None
    assert additional_info.get('too_generic')=='0'
    assert norm_info.get('type') == 'candidate_direct_match'
    assert norm_info.get('score') > 0.99

    norm, additional_info, notes, norm_info = normalizer.normalize('good worker')
    assert notes is not None
    assert notes.get('observations') is not None
    assert norm is not None
    assert additional_info.get('possible_domains') is not None
    assert additional_info.get('possible_fields') is not None
    assert additional_info.get('too_generic')=='1'
    assert norm_info.get('type') == 'new_normalized_string'
    assert norm_info.get('score') is None

    response, distance =  normalizer.vector_search_on_collection(string_to_search= 'test', collection_name = 'verified', limit=10)
    prompt = normalizer.create_prompt("test", response)
    assert prompt[:99] == 'skill_raw;observations;skill_normalized;possible_domains;possible_fields;type_of_skill;too_generic\n'

    utility.drop_collection("new_skill_idx_verified")
    utility.drop_collection("new_skill_idx_candidates")
    utility.drop_collection("cache_raw_embedding")





