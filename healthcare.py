 
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
import json
from pushbullet import PushBullet

client = MongoClient(open('mongodb_key.txt','r').read())
try:
    client.admin.command('ping')
    print("Deployment Successful")
except Exception as e:
    print(e)

pushbullet_api_key=open('pushbullet_api_key.txt','r').read()
pb=PushBullet(pushbullet_api_key)

OPEN_API_KEY=open('apikey.txt','r').read()

list_of_devices=[]
for device in pb.devices:
    name= str(device).split("'",-1)[1]
    list_of_devices.append(name)
print(list_of_devices)
DB_NAME = 'healthcare'
db = client[DB_NAME]

def fillCollection():

    with open('employees.json', 'r') as file:
        employee = json.load(file)

    EMPLOYEE_COLLECTION_NAME = 'employees'
    employee_collection = db[EMPLOYEE_COLLECTION_NAME]

    employee_collection.insert_many(employee)


    with open('h2.json', 'r') as file:
        hospital = json.load(file)


    HOSPITAL_COLLECTION_NAME = 'hospital'
    hospital_collection = db[HOSPITAL_COLLECTION_NAME]

    hospital_collection.insert_many(hospital)


    employee_cursor = employee_collection.find({})
    employee_list = list(employee_cursor)

    d=dict()
    l=list(employee_collection.find({}))
    for entry in l:
        d.update({entry['name']:1})


    hospital_cursor = hospital_collection.find({})
    hospital_list = list(hospital_cursor)


    def concatenate_fields_with_names(document, parent_key=''):
        concatenated_string = ""
        for key, value in document.items():
            if key == '_id':
                continue
            if key == 'full_embedding' or key=='is_available':
                continue
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                concatenated_string += " " + concatenate_fields_with_names(value, full_key)
            else:
                concatenated_string += f" {full_key}:{value}"
        return concatenated_string.strip()

    emp_attributes = []
    for document in employee_list:
        concatenated_string = concatenate_fields_with_names(document)
        emp_attributes.append(concatenated_string)


    hsp_attributes = []
    for document in hospital_list:
        concatenated_string = concatenate_fields_with_names(document)
        hsp_attributes.append(concatenated_string)
    
    return emp_attributes, hsp_attributes, d

def index_prompt(emp_attributes, hsp_attributes):

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small" , openai_api_key=OPEN_API_KEY)

    emp_vec_search = MongoDBAtlasVectorSearch.from_texts(
        texts=emp_attributes,
        embedding=embeddings_model,
        collection=db['emp'],
        index_name="emp_index"
    )


    hsp_vec_search = MongoDBAtlasVectorSearch.from_texts(
        texts=hsp_attributes,
        embedding=embeddings_model,
        collection=db['hsp'],
        index_name="hsp_index"    
    )


    emp_retriever = emp_vec_search.as_retriever(
                            search_type = "similarity",
                            search_kwargs = {"k": 10}
    )


    hsp_retriever = hsp_vec_search.as_retriever(
                            search_type = "similarity",
                            search_kwargs = {"k": 3}
    )

    return emp_retriever, hsp_retriever


def process_prompt(prompt, emp_retriever, hsp_retriever, d):

    model = ChatOpenAI(openai_api_key=OPEN_API_KEY, 
            model_name = 'gpt-4o',
            temperature=0)
    
    availability_dict_json = json.dumps(d)
    availability_dict_json
    
    class StaticRunnable(RunnableLambda):
        def __init__(self, value):
            self.value = value
        def invoke(self, *args, **kwargs):
            return self.value

    
    template3 = """ Your role is to identify the number/amount of people required for a particular task in a hospital 
    from the given prompt. Reply only with a single quantitive word, nothing more. A few examples have been given below -
    Prompt - A patient is to be bandaged in room 201. Answer - One
    Prompt - Three people involved in an accident are in need of aid the ICU. Answer - Three
    Prompt - There has been a huge bus crash. All survivors are being rushed to the hospital. Answer - All
    The prompt is as follows: {question}.
    """
    prompt3 = PromptTemplate.from_template(template = template3, input_vars = ["question"])
    output_parser3 = StrOutputParser()

    retrieval_chain3 = (
        {"question": RunnablePassthrough()}
        | prompt3
        | model
        | output_parser3
    )

    response3 = retrieval_chain3.invoke(prompt)

    template4 = """ Your role is to identify the most appropriate occupation of the people required for a particular task 
    in a hospital from the given prompt. The list of occupations are as follows - Doctor, Receptionist, Nurse,
    Surgical Nurse, Oncology Nurse, Neonatal Nurse, Emergency Nurse, Pediatric Nurse, Surgeon, Orthopedic Surgeon, 
    Surgical Technician, Radiologist, Cook, Oncologist, Anesthesiologist, Pediatrician, Cardiologist, Midwife, 
    Lab Technician, General Practitioner, Neonatologist, Paramedic, Dietitian, Obstetrician, Pathologist, Physiotherapist, 
    Radiographer, Pharmacist, Physician Assistant, Respiratory Therapist, Maintenance Worker, Administrator, 
    Conference Coordinator, IT Specialist, HR Manager, Trainer, Maintenance Manager.
    You can only pick from this list. Give the 3 most appropriate occupations as the only output, with commas in between.
    The prompt is as follows: {question}.
    """
    prompt4 = PromptTemplate.from_template(template = template4, input_vars = ["question"])
    output_parser4 = StrOutputParser()

    retrieval_chain4 = (
        {"question": RunnablePassthrough()}
        | prompt4
        | model
        | output_parser4
    )

    response4 = retrieval_chain4.invoke(prompt)

    template1 = """ Your role is to identify the employees that fall into the job category "{occupation}" mentioned
    using the list of employees: {employee}.
    The availability details is in {dic}. Use this dictionary to identify available employees.
    1 indicates available and 0 indicates not available.You are supposed to find only available employees.
    List only the chosen employees full names and their attributes, nothing more.
    """
    prompt1 = PromptTemplate.from_template(template = template1, input_vars = ["employee", "occupation","dic"])
    output_parser1 = StrOutputParser()
    
    retrieval_chain1 = (
        {"employee": emp_retriever, "occupation": StaticRunnable(response4),"dic": StaticRunnable(availability_dict_json)}
        | prompt1 
        | model 
        | output_parser1
    )


    response1 = retrieval_chain1.invoke(prompt)

    
    template2 = """ 
    From the employees provided in the context, pick the closest employees to the location mentioned in the request/task. 
    The number of employees should be picked depending on the quantitative description "{number}".
    For example - if "one" is the word, pick one employee. If it's "three", pick the three most appropriate employees.
    If it's "multiple", approximate the most appropriate number. If it's "all", take all employees.
    The location is given in the form of x and y co-ordinates. 
    Person on the same floor takes more preference than distance between the locations. 
    If they are at the same location, the one with more experience takes precedence.
    There are a lot of nurses so please assign them properly, there is no way you can run short on the nurses available.
    The output should have only 2 lines, first line with the names of employees seperated with commas and the second with the task they are supposed to perform in a short sentence, not just blatantly saying the task name and location.
    Also the location name is enough, no need for the coordinates in the reply you give.
    The context is {question}
    The locations of all the rooms are given in {room}
    """
    prompt2 = PromptTemplate.from_template(template = template2, input_vars = ["room", "question", "number"])
    output_parser2 = StrOutputParser()

    
    retrieval_chain2 = (
        {"room": hsp_retriever, "question": RunnablePassthrough(), "number": StaticRunnable(response3)}
        | prompt2
        | model 
        | output_parser2
    )

    response2 = retrieval_chain2.invoke(response1)
    
    # print(f"response 3: {response3}\n")
    # print(f"response 4: {response4}\n")
    # print(f"response 1: {response1}\n")
    # print(f"response 2: {response2}\n")

    db.emp.delete_many({})
    db.hsp.delete_many({})
    
    result=response2.splitlines()
    names=result[0].strip().split(",")
    task=result[1][6:]    
    
    for i in names:
        d[i]=0
    
    for name in names:
        if name in list_of_devices:
            device=pb.devices[list_of_devices.index(name)]
            device.push_note(f"{name}\nTask Assigned",task.strip().upper())
    
    return names,task
 
