from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from pymilvus import utility
import pandas as pd
import openai
from dotenv import load_dotenv
import os
from concurrent.futures import TimeoutError
import concurrent.futures
import tqdm
import yaml
# from rwkv.model import RWKV
# from rwkv.utils import PIPELINE, PIPELINE_ARGS
import numba
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_NAME_LLM = os.getenv("MODEL_NAME_LLM", config['MODEL_NAME_LLM'])

ALIAS = os.getenv('ALIAS', config['ALIAS'])
MILVUS_HOST = os.getenv('MILVUS_HOST', config['MILVUS_HOST'])
MILVUS_PORT = os.getenv('MILVUS_PORT', config['MILVUS_PORT'])
# MILVUS_TOKEN = os.getenv('MILVUS_TOKEN', config['MILVUS_TOKEN'])   
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', config['LOGGING_LEVEL'])

N_FOR_PROMPT = int(os.getenv('N_FOR_PROMPT', config['N_FOR_PROMPT']))
TRESHOLD_DIRECT_ACCEPT = float(os.getenv('TRESHOLD_DIRECT_ACCEPT', config['TRESHOLD_DIRECT_ACCEPT']))
TRESHOLD_CANDIDATE_ACCEPT = float(os.getenv('TRESHOLD_CANDIDATE_ACCEPT', config['TRESHOLD_CANDIDATE_ACCEPT']))

DEVICE = os.getenv('DEVICE', config['DEVICE'])

COSISTENCY_LEVEL = os.getenv('COSISTENCY_LEVEL', config['COSISTENCY_LEVEL'])

os.environ['RWKV_JIT_ON'] = '1'

max_retries = 6

# Set up logging
# logging.basicConfig(filename='normalizer.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=getattr(logging, LOGGING_LEVEL))
logger = logging.getLogger(__name__)

connections.connect(
  alias=ALIAS,
  uri=f"{MILVUS_HOST}:{MILVUS_PORT}",
#   token=MILVUS_TOKEN,
)

# download models: https://huggingface.co/BlinkDL
# LLM_MODEL = RWKV(model='models/RWKV-5-World-3B-v2-20231113-ctx4096.pth', strategy='mps fp32')

EMBEDDING_MODEL = SentenceTransformer("thenlper/gte-base").to(DEVICE)

@numba.jit(nopython=True)
def preprocessing(word: str)->str:
    """
    This function preprocesses the input word so that is compatible with Milvus.

    Args:
    ----
        * word (str): 
            The word to be preprocessed.

    Returns:
    ----
        * word (str): 
            The preprocessed word.
    """
    # ecape characters \n and \t " and '
    word = word.replace("\n", "\\n")
    word = word.replace("\t", "\\t")
    word = word.replace('"', '\\"')
    return word
# calling the function once to compile it
# so that it can be used in parallel
preprocessing("word")

class Normalizer():
    """
    This is the Normalizer class. It is used to normalize various types of data. 
    It has several methods for preprocessing and getting word embeddings. 
    It also has methods for getting or creating indices and initializing the normalizer.
    """
    def __init__(self, norm_type: str,
                 threshold_verified_accept: float = TRESHOLD_DIRECT_ACCEPT ,
                 threshold_candidate_accept: float = TRESHOLD_CANDIDATE_ACCEPT,
                 header_raw_string: str = "raw_string",
                 header_norm_string: str = "norm_string",
                 header_notes:List[str] = None,
                 header_additional_info: List[str] = None,
                 n_for_prompt: int = N_FOR_PROMPT):
        """
        Initializes the Normalizer class.

        Args:
        ----
            * norm_type (str): 
                The type of normalization to be performed, this will be used to name the indices.
            * threshold_verified_accept (float, optional):
                The threshold for verified acceptance. Defaults to TRESHOLD_DIRECT_ACCEPT. 
                this is used to avoid a call to the LLM if the string is already in the verified index.
            * threshold_candidate_accept (float, optional): 
                The threshold for candidate acceptance. Defaults to TRESHOLD_CANDIDATE_ACCEPT.
                this is used to avoid a call to the LLM if the string is already in the candidates index.
            * header_raw_string (str, optional): 
                The header for the raw string. Defaults to "raw_string".
                this is the name of the raw string in the csv file used to create the prompt for the LLM.
            * header_norm_string (str, optional): 
                The header for the normalized string. Defaults to "norm_string".
                this is the name of the normalized string in the csv file used to create the prompt for the LLM.
            * header_notes (List[str], optional): 
                The header for the notes. Defaults to None.
                this are the names of the notes in the csv file used to create the prompt for the LLM. 
                The notes are used to add a chain of thought prompting to the normalization of the strings.
                They can be different for even for the same normalized string. Since the notes depend on the raw string.
            * header_additional_info (List[str], optional): 
                The header for additional information. Defaults to None.
                this are the names of the additional information in the csv file used to create the prompt for the LLM.
                The additional information are variables that are associated to the normalized string. 
                Therefore they are the same for the same normalized string.
            * n_for_prompt (int, optional):
                The number of entries to be used from the verified index to create the prompt for the LLM. Defaults to N_FOR_PROMPT.
        """
        self.n_for_prompt = n_for_prompt

        self.norm_type = norm_type
        self.verified_index = norm_type+"_idx_verified"
        self.candidates_index = norm_type+"_idx_candidates"

        self.header_raw_string = header_raw_string
        self.header_norm_string = header_norm_string
        self.header_notes = header_notes
        self.header_additional_info = header_additional_info

        if header_notes is not None:
            self.default_notes = {k: "" for k in header_notes}
        else:
            self.default_notes = None

        self.threshold_verified_accept = threshold_verified_accept
        self.threshold_candidate_accept = threshold_candidate_accept

        logger.info(f"Normalizer initialized with type {self.norm_type}.")

        self.collection_verified, self.collection_candidates = self.get_or_create_indices()


    def get_word_embedding(self, words)->dict:
        """
        This function retrieves the word embedding for the input word.

        Args:
        ----
            * words (str or list of str): 
                The word or list of words for which the embedding is to be retrieved.

        Returns:
        ----
            * embedding_dict (dict): 
                The embedding of the input word.
        """

        if isinstance(words, str):
            words = [words]

        words = [preprocessing(word) for word in words]
        embeddings = EMBEDDING_MODEL.encode(words)

        return {'words': words, 'embeddings': embeddings}

    def cosine_similarity(self, input1, input2)->float:
        """
        This function calculates the cosine similarity between two inputs. The inputs can be either strings or vectors.
        If the inputs are strings, they are converted to vectors using the get_word_embedding method.
        
        Args:
            input1 (str or np.array): The first input. Can be a string or a vector.
            input2 (str or np.array): The second input. Can be a string or a vector.
        
        Returns:
            float: The cosine similarity between the two inputs.
        """
        # Convert words to vectors if necessary
        vector1 = self.get_word_embedding(input1)['embeddings'][0] if isinstance(input1, str) else input1
        vector2 = self.get_word_embedding(input2)['embeddings'][0] if isinstance(input2, str) else input2

        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def get_or_create_collection(self, collection_name, description):
        """
        This function gets or creates a collection in Milvus. If the collection already exists, it is returned. 
        If it does not exist, it is created with the specified index name and description, and then returned.

        Args:
            collection_name (str): The name of the index.
            description (str): The description of the index.

        Returns:
            Collection: The collection with the specified index name that will be used for the normalizer.
        """

        if utility.has_collection(collection_name):
            logger.info(f"Index {collection_name} already exists.")
            collection = Collection(collection_name)
            
        else:
            raw_string = FieldSchema(
                name="raw_string",
                dtype=DataType.VARCHAR,
                max_length=200,
                is_primary=True)
            
            notes = FieldSchema(
                name="notes",
                dtype=DataType.JSON,
            )

            norm_string = FieldSchema(
                name="norm_string",
                dtype=DataType.VARCHAR,
                max_length=200,
                default_value="Unknown"
            )
            additional_info = FieldSchema(
                name="additional_info",
                dtype=DataType.JSON
            )
            raw_string_embedding = FieldSchema(
                name="raw_string_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=768
            )

            schema = CollectionSchema(
                fields=[raw_string, notes, norm_string, additional_info, raw_string_embedding],
                description=description,
                enable_dynamic_field=True
            )
            collection = Collection(
                name=collection_name,
                schema=schema,
                using='default',
                shards_num=2,
                consistency_level=COSISTENCY_LEVEL,
            )

        if len(utility.list_indexes(collection_name))<2:
            
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {
                    "nlist": 16384,
                    "nprobe": 16,
                }
            }
            
            status = collection.create_index(field_name="raw_string_embedding", index_params=index_params)
            logger.info(f"Index {collection_name} created. Status: {status}")
            status = collection.create_index(field_name="norm_string", index_name=f"{collection_name}_norm_string_index")
            logger.info(f"norm_string index cache_raw_embedding created. Status: {status}")

        return collection

    def get_or_create_indices(self):
        """
        This function is used to get or create indices for the collections that the normalizer will use. 

        It first generates descriptions for the verified and candidates collections. 
        Then it calls the get_or_create_collection method for each of these collections 
        and stores the returned collections in the variables collection_verified and collection_candidates.


        The verified collection will contain the normalized strings that have been verified by an expert,
        for now these will be added via the massive_add_to_index method and a CSV file.

        The candidates collection will contain the normalized strings that have been generated by the LLM,
        and that have not been verified yet.

        Returns:
        ----
            * collection_verified (Collection): 
                The verified collection.
            * collection_candidates (Collection): 
                The candidates collection.
        """
        description_verified = f"{self.norm_type} verified collection"
        description_candidates = f"{self.norm_type} candidates collection"
        
        collection_verified = self.get_or_create_collection(self.verified_index, description_verified)
        collection_candidates = self.get_or_create_collection(self.candidates_index, description_candidates)
        
        return collection_verified, collection_candidates
    
    def convert_to_milvus_format(self, row):
        """
        This function is used to convert a row from a CSV file to the format that Milvus expects.

        Args:
        ----
            * row (pd.Series): 
                The row from the CSV file.
        
        Returns:
        ----
            * raw_string (str): 
                The raw string.
            * notes (dict): 
                The notes. The keys are the names given in the header_notes and the values are the values in the row.
            * norm_string (str): 
                The normalized string.
            * additional_info (dict): 
                The additional information. The keys are the names given in the header_additional_info and the values are the values in the row.
        """
        
        raw_string = str(row[self.header_raw_string])
        notes = {k: str(row[k]) for k in self.header_notes} if self.header_notes is not None else None
        norm_string = str(row[self.header_norm_string])
        additional_info = { k: str(row[k]) for k in self.header_additional_info} if self.header_additional_info is not None else None
        return raw_string, notes, norm_string, additional_info
    
    def massive_add_to_index(self, csv_file_path: str):
        """
        This function is used to add a CSV file to the indices.
        This is usefull when the indices are empty and we want to add a large number of strings to them.

        Args:
        ----
            * csv_file_path (str): 
                The path to the CSV file.
        """

        df = pd.read_csv(csv_file_path, header=0, sep=';')

        if self.collection_candidates.num_entities >= 2*len(df) and self.collection_verified.num_entities == 2*len(df):
            logger.info(f"Index {self.collection_candidates.name} already contains all the rows in the csv file.")
            return
        elif self.collection_candidates.num_entities > 0 or self.collection_verified.num_entities > 0:
            logger.warning(f"""Index {self.collection_candidates.name} or {self.collection_verified.name} already contains some rows.
                            But the numbers of rows in the csv file is different. a check is needed.
                            We will proceed with the upsert of the csv file.""")
        
        list_of_raws = []
        list_of_notes = []
        list_of_norms = []
        list_of_additional_infos = []
        list_of_raw_embeddings= self.get_word_embedding(df[self.header_raw_string].tolist())['embeddings']
        # list_of_raw_embeddings_norm = self.get_word_embedding(df[self.header_norm_string].tolist())['embeddings']

        # list_of_raw_embeddings = [val for pair in zip(list_of_raw_embeddings_raw, list_of_raw_embeddings_norm) for val in pair]

        for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            raw_string, notes, norm_string, additional_info = self.convert_to_milvus_format(row)

            list_of_raws.append(raw_string)
            list_of_norms.append(norm_string)
            list_of_notes.append(notes)
            list_of_additional_infos.append(additional_info)

            # list_of_raws.append(norm_string)
            # list_of_norms.append(norm_string)
            # list_of_notes.append(notes)
            # list_of_additional_infos.append(additional_info)

        list_of_lists_for_upsert = [list_of_raws, list_of_notes, list_of_norms, list_of_additional_infos, list_of_raw_embeddings]

        self.collection_candidates.upsert(list_of_lists_for_upsert)
        self.collection_verified.upsert(list_of_lists_for_upsert)

        self.collection_candidates.flush()
        self.collection_verified.flush()

        self.collection_candidates.load()
        self.collection_verified.load()
            

    def add_to_index(self, collection_name: str,
                    raw_string: str,
                    norm_string: str,
                    additional_info: dict = None,
                    notes: dict = None,
                    embedding: np.array = None,
                    flush_db: bool = False):
        """
        This function is used to add a raw string to the index. If the embedding is not provided, it is calculated using the get_word_embedding method.

        Args:
        ----
            * collection_name (str):
                The name of the collection to which the raw string should be added.
            * raw_string (str):
                The raw string to be added.
            * norm_string (str):
                The normalized string to be added.
            * additional_info (dict, optional):
                The additional information to be added. Defaults to None.
            * notes (dict, optional):
                The notes to be added. Defaults to None.
            * embedding (np.array, optional):
                The embedding of the raw string. Defaults to None.
            * flush_db (bool, optional):    
                Whether to flush the database after adding the raw string. Defaults to False.
        """

        if self.header_notes is not None and notes is None:
            raise ValueError(f"""These notes are required for this index: {self.header_notes}
                              Please provide them.""")
        if self.header_notes is None and notes is not None:
            raise ValueError("Notes are not required for this index. Please remove them.")
        if self.header_additional_info is not None and additional_info is None:
            raise ValueError(f"""These additional info is required for this index: {self.header_additional_info}
                              Please provide them.""")
        if self.header_additional_info is None and additional_info is not None:
            raise ValueError("Additional info is not required for this index. Please remove it.")
        

        if embedding is None:
            embedding = self.get_word_embedding(raw_string)['embeddings'][0]

        if collection_name == 'candidates':
            self.collection_candidates.upsert([[raw_string],
                           [notes],
                           [norm_string],
                           [additional_info],
                           [embedding]])

        elif collection_name == 'verified':
            self.collection_candidates.upsert([[raw_string],
                           [notes],
                           [norm_string],
                           [additional_info],
                           [embedding]])
            self.collection_verified.upsert([[raw_string],
                           [notes],
                           [norm_string],
                           [additional_info],
                           [embedding]])
            
        if flush_db:
            self.collection_candidates.flush()
            self.collection_verified.flush()
            self.collection_candidates.load()
            self.collection_verified.load()

    def check_if_normalized_is_in_collection(self, norm_string: str, collection_name: str='candidates'):
        """
        This function is used to search if a normalized string is in the index.

        Args:
        ----
            * norm_string (str):
                The normalized string to be checked.
            * collection_name (str, optional):
                The name of the collection to be checked. Defaults to 'candidates'.

        Returns:
        ----
            * bool:
                Whether the normalized string is in the collection.
            * result:
                The result of the query.
        """
        if collection_name == 'candidates':
            collection = self.collection_candidates
        elif collection_name == 'verified':
            collection = self.collection_verified
        else:
            raise ValueError(f"Collection name {collection_name} is not valid.")
        
        if str(utility.load_state(collection.name))=='NotLoad':
            logger.info(f"Loading collection {collection.name}")
            collection.load()

        norm_string = preprocessing(norm_string)
        try:
            res = collection.query(
                    expr = f"norm_string == '{norm_string}'",
                    offset = 0,
                    limit = 1, 
                    output_fields = ["raw_string", "norm_string","additional_info", "notes"]
                    # cosistency_level=COSISTENCY_LEVEL
                    )
        except Exception as e:
            logger.error(f"""Error while querying the collection for normalized string:
                          {norm_string}.""")
            logger.error(e)
            return False, None
        if len(res) == 0:
            return False, None
        else:
            return True, res[0]        

    def vector_search_on_collection(self,
                                    string_to_search: str=None,
                                    embedding: np.array=None,
                                    collection_name: str='verified',
                                    limit: int=1)-> Tuple[list, list]:
        """
        This function is used to performa a vector search for a string or an embedding in the index.
        Only one of string_to_search or embedding should be provided.
        
        Args:
        ----
            * string_to_search (str, optional):
                The string to be searched. Defaults to None.
            * embedding (np.array, optional):
                The embedding to be searched. Defaults to None.
            * collection_name (str, optional):
                The name of the collection to be searched. Defaults to 'verified'.
            * limit (int, optional):
                The limit of the search. Defaults to 1.
        
        Returns:
        ----
            * results (list):
                The results of the search.
            * distances (list):
                The distances of the search.

        """

        if string_to_search is not None and embedding is not None:
            raise ValueError("Both string_to_search and embedding were provided. Only one should be provided.")
        elif string_to_search is None and embedding is None:
            raise ValueError("Neither string_to_search nor embedding were provided. One should be provided.")
        
        if string_to_search is not None:
            embedding = self.get_word_embedding(string_to_search)['embeddings'][0]

        if collection_name == 'candidates':
            collection = self.collection_candidates
        elif collection_name == 'verified':
            collection = self.collection_verified
        else:
            raise ValueError(f"Collection name {collection_name} is not valid.")

        search_params = {
        "metric_type": "IP",
        "offset": 0, 
        "ignore_growing": False, 
        }

        if str(utility.load_state(collection.name))=='NotLoad':
            logger.info(f"Loading collection {collection.name}")
            collection.load()

        res = collection.search(data=[embedding], 
            anns_field="raw_string_embedding", 
            param=search_params,
            limit=limit,
            output_fields=['raw_string', 'norm_string', 'additional_info', 'notes'],
            consistency_level=COSISTENCY_LEVEL)[0]

        return [hit.fields for hit in res][::-1], res.distances[::-1]

    def create_prompt(self, raw_string, verified_results, fixed_part_path=None):
        """
        This function is used to create a prompt for the LLM. The prompt is created using the raw string and the data in the verified index.
        And is composed of:
            * header: 
                The header of the csv file used to create the prompt. 
                Created from the header_raw_string, header_notes, header_norm_string and header_additional_info.
            * fixed_part:
                The fixed part of the prompt. Read from the fixed_part_path.
            * variable_part:
                The variable part of the prompt. Created from the data in the verified index.
            * raw_string:
                The raw string for which the prompt is created.

        Args:
        ----
            * raw_string (str):
                The raw string for which the prompt is to be created.
            * verified_results (list):
                The data in the verified index taken from the vector_search_on_collection method.
                This is used to create the prompt for the LLM.
            * fixed_part_path (str, optional):
                The path to the fixed part of the prompt. Defaults to None.
        
        Returns:
        ----
            * prompt (str):
                The prompt for the LLM.
        """

        list_for_csv = [ ]
        for d in verified_results:
            dict_for_row = {'raw_string': d['raw_string'], 'norm_string': d['norm_string']}
            if self.header_notes is not None:
                dict_for_row = {'raw_string': dict_for_row['raw_string'], **d['notes'], 'norm_string': dict_for_row['norm_string']}
            if self.header_additional_info is not None:
                dict_for_row = {**dict_for_row, **d['additional_info']}
            list_for_csv.append(dict_for_row)
        df_for_prompt = pd.DataFrame(list_for_csv)
        variable_part = df_for_prompt.to_csv(index=False, sep=';', header=False)
        
        header_parts = [self.header_raw_string]
        if self.header_notes is not None:
            header_parts += self.header_notes
        header_parts.append(self.header_norm_string)
        if self.header_additional_info is not None:
            header_parts += self.header_additional_info
        header = ";".join(header_parts)+'\n'

        if fixed_part_path is not None:
            fixed_part = open(fixed_part_path, 'r').read()
            prompt = header+\
                fixed_part+\
                ('' if fixed_part.endswith('\n') else '\n')+\
                variable_part+raw_string+';'
        else:
            prompt = header+variable_part+raw_string+';'
        return prompt

    def call_api_with_timeout(self, prompt):
        risp = openai.Completion.create(
                            model=MODEL_NAME_LLM,
                            prompt=prompt,
                            temperature=0.0,
                            stop=["\n"],
                            max_tokens=100,)
        return risp.choices[0]['text']
    
    # def call_rwkv(self, prompt):
    #     """this function is used to call the RWKV to normalize a string."""

    #     pipeline = PIPELINE(LLM_MODEL, "rwkv_vocab_v20230424") #for rwkv "world" models

    #     args = PIPELINE_ARGS(temperature = 0, 
    #                         token_stop = [11]) 
   
    #     return pipeline.generate(prompt, args=args)

    def LLM_call(self, raw_string: str, verified_results: list):
        """
        This function is used to call the LLM to normalize a string.

        Args:
        ----
            * raw_string (str):
                The raw string to be normalized.
            * verified_results (list):
                The data in the verified index taken from the vector_search_on_collection method.
                This is used to create the prompt for the LLM.
        
        Returns:
        ----
            * norm_string (str):
                The normalized string.
            * additional_info (dict, optional):
                The additional information associated with the normalized string.
            * notes (dict, optional):
                The notes associated with the raw string. Created by the LLM to normalize the string.
        """
            # load csv_prompt
        prompt = self.create_prompt(raw_string, verified_results)

        llm_call_failed = True
        for i in range(max_retries):
            try:
                resp = self.call_api_with_timeout(prompt)
                llm_call_failed = False
                break
            except Exception as e:
                logging.warn(f"error: {e}")
                logging.warn(f"API call timed out. Attempt {i+1} of {max_retries}.")
                time.sleep(2 ** i)  # exponential backoff
        if llm_call_failed:
            raise TimeoutError("API call timed out after maximum number of retries.")
        else:
            logging.info(f"API call successful after {i+1} attempts.")
            try:
                list_response_llm = resp.split(';')
                norm_position = 0

                if self.header_notes is not None:
                    notes = {k:v for k,v in zip(self.header_notes, list_response_llm[:len(self.header_notes)])}
                    norm_position = len(self.header_notes)
                else:
                    notes = None
                
                norm_string = list_response_llm[norm_position]

                if self.header_additional_info is not None:
                    additional_info = {k:v for k,v in zip(self.header_additional_info, list_response_llm[norm_position+1:])}
                else:
                    additional_info = None

                return norm_string, additional_info, notes
            except Exception as e:
                logging.error(f"""
                --- LLM response not valid. ---
                -> Response: {resp}
                -> Prompt: {prompt}
                -> Error: {e}
                """)
                return None, None, None

    def normalize(self, raw_string,
                threshold_verified_accept=None,
                threshold_candidate_accept=None,
                flush_db=False):
        """
        This function is used to normalize a raw string. It checks if the raw string is already in the verified or candidates collection and if not, it uses the LLM to normalize the string.

        To avoid calls to the LLM, you can adjust the threshold values:
            * Set threshold_verified_accept to >1 and threshold_candidate_accept <0 to only match in the candidates collection. 
            * Set threshold_verified_accept to <0 to only match in the verified collection.

        Args:
        ----
            * raw_string (str): 
                The raw string to be normalized.
            * threshold_verified_accept (float, optional):
                The threshold for accepting a match in the verified collection. Defaults to self.threshold_verified_accept.
            * threshold_candidate_accept (float, optional):
                The threshold for accepting a match in the candidates collection. Defaults to self.threshold_candidate_accept.
            * flush_db (bool, optional):
                If set to True, it will flush the database after adding a new entry. Defaults to False.

        Returns:
        ----
            * norm_string (str): 
                The normalized string.
            * additional_info (dict, optional):
                The additional information associated with the normalized string.
            * notes (dict, optional):
                The notes associated with the raw string. Created by the LLM to normalize the string.
            * normalization_info (dict):
                Information about the normalization process, including the type of match found and the score.
        """

        if threshold_verified_accept is None:
            threshold_verified_accept = self.threshold_verified_accept
        if threshold_candidate_accept is None:
            threshold_candidate_accept = self.threshold_candidate_accept

        normalization_info = {"type": None, "score": None}
        # get raw string embedding
        logger.info(f"Normalizing {raw_string}")
        if len(raw_string) > 200:
            logger.warning(f"Raw string {raw_string} is too long. Truncating it.")
            raw_string = raw_string[:200]
        raw_embedding = self.get_word_embedding(raw_string)['embeddings'][0]

        # check if raw string is already in the verified index using vector search
        verified_results, verified_distances = self.vector_search_on_collection(embedding=raw_embedding, collection_name='verified', limit=self.n_for_prompt)
        if verified_results and verified_distances[-1] >= threshold_verified_accept:
            normalization_info = {"type": "verified_direct_match", "score": verified_distances[-1]}
            logger.info(f"Verified direct match found for {raw_string}. Normalized string: {verified_results[-1]['norm_string']}")
            return verified_results[-1]['norm_string'], verified_results[-1]['additional_info'], verified_results[-1]['notes'], normalization_info

        # if no, check if raw string is already in the candidates index using vector search
        candidates_results, candidates_distances = self.vector_search_on_collection(embedding=raw_embedding, collection_name='candidates', limit=1)
        if candidates_results and candidates_distances[-1] >= threshold_candidate_accept:
            normalization_info = {"type": "candidate_direct_match", "score": candidates_distances[-1]}
            logger.info(f"Candidate direct match found for {raw_string}. Normalized string: {candidates_results[-1]['norm_string']}")
            return candidates_results[-1]['norm_string'], candidates_results[-1]['additional_info'], candidates_results[-1]['notes'], normalization_info

        # if no, use LLM to normalize the string
        logger.info(f"No direct match found for {raw_string}. Calling LLM.")
        norm_string, additional_info_llm, notes = self.LLM_call(raw_string, verified_results)
        logger.info(f"""LLM response for {raw_string}: {norm_string},
                        additional_info: {additional_info_llm},
                        notes: {notes}""")

        # check if normalized string is already in the candidates index
        in_candidates, candidates_result = self.check_if_normalized_is_in_collection(norm_string, 'candidates')
        logger.info(f"Normalized string {norm_string} is in candidates: {in_candidates}")
        if in_candidates:
            # overwrite the new associated info with the old associated info
            additional_info = candidates_result['additional_info']
            logger.info(f"Normalized string {norm_string} is in candidates. Overwriting the LLM output. With additional info taken from db: {additional_info}")
            normalization_info = {"type": "candidate_indirect_match", "score": None}
        else:
            additional_info = additional_info_llm
            normalization_info = {"type": "new_normalized_string", "score": None}
            # logger.info(f"Normalized string {norm_string} is not in candidates. Adding it. With additional info: {additional_info}")
            # we add to the candidates index the normalized string as if it was a new raw string
            # self.add_to_index('candidates', norm_string, norm_string, additional_info, self.default_notes, self.get_word_embedding(norm_string)['embeddings'][0], flush_db)

        # in any case, add the couple (raw string, normalized string+associated info) to the candidates index
        self.add_to_index('candidates', raw_string, norm_string, additional_info, notes, raw_embedding, flush_db)

        return norm_string, additional_info, notes, normalization_info


class SkillNormalizerOld(Normalizer):
    def __init__(self, test=""):
        super().__init__("skill"+test,
                         header_raw_string="skill_raw",
                         header_norm_string="skill_normalized",
                         header_additional_info=["domain", "field", "type_of_skill"])
        

    def massive_add_to_index(self):
        return super().massive_add_to_index('data/skills_dataset.csv')
    
class JobNormalizer(Normalizer):
    def __init__(self, test=""):
        super().__init__("job"+test, 
                         header_raw_string="declared_job_role",
                         header_norm_string="normalized_job_role",
                         header_additional_info=["general_cluster", "function"],
                         n_for_prompt=15)

    def massive_add_to_index(self):
        return super().massive_add_to_index('data/job_roles_dataset.csv')
    
    def create_prompt(self, raw_string, verified_results):
        return super().create_prompt(raw_string, verified_results, 'data/job_roles_dataset_prompt.csv')
    
class JobNormalizerWithHierarchy(Normalizer):
    # declared_job_role;hierarchical_position;normalized_job_role;general_cluster;function
    def __init__(self, test=""):
        super().__init__("job_with_hierarchy"+test, 
                         header_raw_string="declared_job_role",
                         header_norm_string="normalized_job_role",
                         header_notes=['hierarchical_position'],
                         header_additional_info=["general_cluster", "function"],
                         n_for_prompt=15)

    def massive_add_to_index(self):
        return super().massive_add_to_index('data/job_roles_with_hierarchical_position.csv')
    
    def create_prompt(self, raw_string, verified_results):
        return super().create_prompt(raw_string, verified_results, 'data/job_roles_with_hierarchical_position_prompt.csv')
    
    def call_api_with_timeout(self, prompt):
        risp = openai.Completion.create(
                            model=MODEL_NAME_LLM,
                            prompt=prompt,
                            temperature=0.0,
                            stop=["\n"],
                            max_tokens=100,
                            logprobs=5)

        dict_acceptable_hierarchy_level = {'Special':'Specialist',
                                          'Manager':'Manager',
                                          'Team':'Team Leader',
                                          'Junior':'Junior',
                                          'Head':'Head',
                                          'Intern':'Internship'}
        candidates_hierarchy_level = list(risp.choices[0]['logprobs']['top_logprobs'][0])

        hierarchy_level = 'Unknown'
        for candidate_hierarchy_level in candidates_hierarchy_level:
            if candidate_hierarchy_level in dict_acceptable_hierarchy_level:
                hierarchy_level = dict_acceptable_hierarchy_level[candidate_hierarchy_level]
                break
        
        
        final_output = risp.choices[0]['text'].split(';')
        final_output[0] = hierarchy_level
        return ";".join(final_output)


    
class LanguageNormalizer(Normalizer):
    def __init__(self, test=""):
        super().__init__("language"+test, 
                         header_raw_string="language_raw",
                         header_norm_string="language_normalized",
                         header_additional_info=["dead_language", "sign_language"],
                         n_for_prompt=15)
    
    def massive_add_to_index(self):
        return super().massive_add_to_index('data/languages.csv')
    
    def create_prompt(self, raw_string, verified_results):
        return super().create_prompt(raw_string, verified_results, 'data/languages_prompt.csv')
    
class SkillNormalizerNew(Normalizer):
    def __init__(self, test=""):
        super().__init__("new_skill"+test, 
                         header_raw_string="skill_raw",
                         header_norm_string="skill_normalized",
                         header_notes=["observations"],
                         header_additional_info=["possible_domains", "possible_fields", "type_of_skill", "too_generic"],
                         n_for_prompt=15)
        
        self.default_notes = {"observations": "no observations"}

    def massive_add_to_index(self):
        return super().massive_add_to_index('data/new_skills_dataset.csv')

    def create_prompt(self, raw_string, verified_results):
        return super().create_prompt(raw_string, verified_results, 'data/new_skills_dataset_prompt.csv')

if __name__ == "__main__":
    normalizer = JobNormalizer()

    # print(normalizer.get_word_embedding("test")))
    print(normalizer.normalize("senior and great data engineer"))

