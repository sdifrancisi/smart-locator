import os
import json
import requests
from langchain.llms import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import Vector
from azure.search.documents import SearchClient
from requests.exceptions import Timeout


from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel
from datetime import datetime, date

# Load environment variables
if load_dotenv():
    print("Found OpenAPI Base Endpoint: " + os.getenv("OPENAI_API_BASE"))
else: 
    print("No file .env found")

openai_api_type = os.getenv("OPENAI_API_TYPE")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
embedding_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
acs_service_name = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME")
acs_endpoint_name = os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT_NAME")
acs_index_name = "index-stores"
acs_api_key = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
print("openai_api_type = " + openai_api_type)
print("Configuration loaded.  ")

# Create an Embeddings Instance of Azure OpenAI
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment=embedding_name,
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key
)
# Create a Completion Instance of Azure OpenAI
llm = AzureChatOpenAI(
    model="gpt-3.5-turbo",
    deployment_name = deployment_name,
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    temperature=0.1,
    max_tokens=500
)

class CompletionRequest(BaseModel):
    Question: str

class CompletionResponse(BaseModel):
    Completion: str
    Stores: dict

def make_nsl_call(request_url: str):
    try:
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}

        response = requests.get(url=request_url,
                                 headers=headers,
                                 timeout=(500, 15000)
                                 )

        json_result = response.json()        
        return json_result

    except Timeout:
        print(f"Timeout during openapi call")

    answer = st.empty()
    url= "https://store-locator.it.edenred.io/v1/dxl/orch-nsl-registry-service/stores?categories=" + results + "&brandCodes=PRE&formats=Digitale&latitude=45.4842675&longitude=9.199799&radius=20&provinceCode=MI&isOnline=false&page=0&size=20&sort=distance&sortDirection=asc&areaCode=TC"
    answer = make_nsl_call(request_url=url, question=result["category_id"])


@tool
def execute_completion(request: CompletionRequest):
    '''Dimmi l'id della categoria di appartenenza del prodotto presente in questa frase: {original_question}
        Per la risposta utilizza solo ed esclusivamente la lista delle categorie qui di seguito dando in risposta la migliore corrispondenza senza mostrare lo score:
        {search_results}

        Rispondi con un json contenente in un attributo denominato category che contiene la categoria completa di ID (category_id) e, in un'altro attributo, una risposta gentile da dare all'utente riguardo alla categoria individuata.
        '''
    # Ask the question
    # The question is being passed in via the message body.
    # request: CompletionRequest
    
    # Create a prompt template with variables, note the curly braces
    prompt = PromptTemplate(
        input_variables=["original_question","search_results"],
        template="""
        Dimmi l'id della categoria di appartenenza del prodotto presente in questa frase: {original_question}
        Per la risposta utilizza solo ed esclusivamente la lista delle categorie qui di seguito dando in risposta la migliore corrispondenza senza mostrare lo score:
        {search_results}

        Rispondi con un json contenente in un attributo la categoria completa di ID e in un'altro attributo una risposta gentile da dare all'utente riguardo alla categoria individuata.
        """,
    )

    # Get Embedding for the original question
    question_embedded=embeddings.embed_query(request.Question)

    # Search Vector Store
    search_client = SearchClient(
        acs_endpoint_name,
        acs_index_name,
        AzureKeyCredential(acs_api_key)
    )


    # You can see here that we are getting the embedding representation of the query.
    vector = Vector(
    value=embeddings.embed_query(request.Question),
    k=3,
    fields="content_vector"
    )
 
    # Execute the search
    results = list(search_client.search(
    search_text="",
    include_total_count=True,
    vectors=[vector],
    select=["category_id", "category_name"],
    ))

       # Print count of total results.
    print(f"Returned {len(results)} results using only vector-based search.")
    print("----------")   
    #   Iterate over results and print out the content.
    for result in results:
        print("SEARCH DB")
        print(result["category_id"])
        print(result["category_name"])
        print(f"Score: {result['@search.score']}")
        #print(result)
        print("----------")

    # Build the Prompt and Execute against the Azure OpenAI to get the completion
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response = chain.run({"original_question": request.Question, "search_results": results})
    print("LLM response: " + response)

    # call store locator api
    jsonResponse = json.loads(response)
    category = jsonResponse["category"]
    
    url= "https://store-locator.it.edenred.io/v1/dxl/orch-nsl-registry-service/stores?categories=" + category["category_id"] + "&brandCodes=PRE&formats=Digitale&latitude=45.4842675&longitude=9.199799&radius=20&provinceCode=MI&isOnline=false&page=0&size=20&sort=distance&sortDirection=asc&areaCode=TC"
    print(url)
    answer = make_nsl_call(request_url=url)    

    return CompletionResponse(Completion = jsonResponse["response"], Stores = answer)