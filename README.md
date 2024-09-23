# HowAbout RAG

## Purpose & Detail

HowAbout의 RAG Application의 핵심 기능은 클라이언트의 요청을 바탕으로 데이팅과 관련된 데이트 활동들을 생생이다.

위 목표를 달성하기 위해 구현에 있어 기본적이 요구를 구축한 다음 고도화할 수 있는 부분들을 추려내어 추가적인 구현을 통해 좀 더 높은 성능의 RAG API Server를 구현했다.
그래서 아래와 같은 부분을 전환하여 고도화가 가능하다고 판단되어 아래의 순서대로 구현을 진행했다.

1. Query Translation: 요청 분석
2. Faiss: 요청과 관련된 내용 조회
3. Prompt Engineering and Parsing: 조회된 내용을 바탕으로 생성 및 클라이언트가 받을 수 있는 형태로의 전환

아래의 내용들은 구현에 대한 구체적인 설명과 HowAbout RAG의 구현 결과를 정리한 내용이다.

## Contributor

<div align="center">

| ![Wooyong Jeong](https://github.com/jwywoo.png?size=300)|
|:-------------------------:|
| [jwywoo26@egmail.com](mailto:jwywoo26@gmail.com) |
| [GitHub](https://github.com/jwywoo) |
| **Wooyong Jeong: Woo**             |
| AI Developer             |

</div>

## Tech Stack

### Environment

![git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)
![v](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![c](https://img.shields.io/badge/Google%20colab-F9AB00?style=for-the-badge&logo=Google%20colab&logoColor=white)
![c](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white)
![c](https://img.shields.io/badge/GitKraken-179287?style=for-the-badge&logo=GitKraken&logoColor=white)
![c](https://img.shields.io/badge/Amazon%20EC2-FF9900?style=for-the-badge&logo=Amazon%20EC2&logoColor=white)
![g](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
![f](https://img.shields.io/badge/Figma-F24E1E?style=for-the-badge&logo=figma&logoColor=white)

### Development

<div>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" />
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=LangChain&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white" />
<img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=OpenAI&logoColor=white" />
<img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=Hugging%20Face&logoColor=black" />
<img src="https://img.shields.io/badge/FAISS-3776AB?style=flat&logo=&logoColor=white"/>
<img src="https://img.shields.io/badge/JSON-000000?style=flat&logo=JSON&logoColor=white"/>
<img src="https://img.shields.io/badge/Selenium-43B02A?style=flat&logo=Selenium&logoColor=white"/>
</div>

## Service Architecture: Howabout RAG Api

<div align="center">

![Service Architecture](/static/project//project_structure.png)
</div>

## Development Process

1. RAG
    - [RAG Practice Github Repository](https://github.com/jwywoo/RAG)
    - AGILE: 최소 작동 단위의 RAG 구현이후 기능별 고도화 진행
        - MVP RAG
        ![MVP RAG](/static/project/rag_mvp.png)
        - MVP RAG += Vector Store
        ![MVP RAG](/static/project/rag_mvp_faiss.png)
        - MVP RAG += Query Translation
        ![MVP RAG](/static/project/rag_mvp_faiss_query.png)
        - MVP RAG += Prompt Template
        ![MVP RAG](/static/project/rag_mvp_faiss_query_prompt.png)

2. RAG API
    - [RAG Service Github Repository](https://github.com/jwywoo/RAG)
    - Turning RAG Practice into API using FastAPI
    - Same as Service Architecture

## Project Structure: Howabout RAG

```vim
HOWABOUT-RAG/
├── .venv
├── .env
├── .gitignore
├── docker_build_push.sh
├── Dockerfile
└── app/
    ├── combined_faiss_index/
    │   ├── index.faiss
    │   └── index.pkl
    ├── core/
    │   └── config.py
    ├── crud/
    │   ├── rag_methods/
    │   │   ├── query_chain.py
    │   │   ├── rag_chain.py
    │   │   ├── result_parsing.py
    │   │   └── retrieval_chain.py
    │   └── dating_crud.py
    ├── prompt_templates/
    │   └── prompt_templates_enum.py
    ├── routers/
    │   └── dating_generation_router.py
    ├── schema/
    │   ├── request_dto/
    │   │   └── dating_generation_request_dto.py
    │   └── response_dto/
    │       └── dating_generation_response_dto.py
    └── main.py
```

## Dating Activity Generation: Woo

### Generation Process

1. Request from Client: `app/schema/request_dto/dating_generation_request_dto.py` and `app/routers/dating_generation_router.py`
    - Routing reqeust to CRUD

    ```Python
    # dating_generation_router.py
    @router.post("/ai/dating/generate", response_model=list[DatingGenResponseDto])
    def dating_generation_router(request: DatingGenRequestDto) -> list[DatingGenRequestDto]:
    return dating_generation(request)
    ```

    - Request DTO for Retrieval and Generation

    ```Python
    # dating_generation_request_dto.py
    class DatingGenRequestDto(BaseModel):
        title: str
        description: str
        dateTime: str
        activityDescription: str
    ```

2. Request Translation & Retrieval: `app/crud/rag_methods/retrieval_chain.py`, `app/crud/rag_methods/query_chain.py`, `app/prompt_templates/prompt_templates_enum.py` and `app/crud/dating_crud.py`
    - Turning request dto to String for query generation

    ```Python
    # dating_crud.py
    def dating_generation(request: DatingGenRequestDto) -> list[DatingGenResponseDto]:
        ...
        start_time = time.time()

        # Question query
        question_query = request.title + SEP + request.description + \
            SEP + request.dateTime + SEP + request.activityDescription

        # Load Vector Store
        # Embedding: OpenAI
        print("embeddings")
        embeddings = load_embeddings(
            model_name="text-embedding-3-large"
        )

        # Templates for query and answer generation
        templates_enum = PromptsEnum
        query_gen_template = templates_enum.query_gen_template.value
        answer_gen_template = templates_enum.answer_gen_template.value

        # Retrieval Chain + Query Analysis
        # for local test: "combined_faiss_index"
        # for deployment: "app/combined_faiss_index"
        print("Retrieval Chain")
        retrieval_chain = get_retrieval_chain(
            template=query_gen_template,
            model_name="gpt-3.5-turbo",
            vector_store_path="combined_faiss_index",
            embeddings=embeddings
        )

        ...
    ```

    - Template for query generation

    ```Python
    # prompt_templates_enum.py
    class PromptsEnum(Enum):
        query_gen_template = """
            You are an expert planning a date for loved one, family and friends.
            Your task is retrieving relevant data to generate a date plan.
            You have access to a database of locations for dating in Seoul.
            Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector
            database.
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.


            Each row in the table represents a location and its features.
            Features are separated by [SEP].
            If a row have 'None' in the feature, it means that the row doesn't have that feature.
            Every row is in Korean while column names are in English.
            Provide these alternative questions separated by newlines.
            Original question: {question}
        """
    ```

    - Query Chain for Query Generation

    ```Python
    # query_chain.py
    def get_query_chain(template, model_name):
        prompt_perspectives = ChatPromptTemplate.from_template(template)
        generate_queries = (
            prompt_perspectives
            | ChatOpenAI(model=model_name, temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        return generate_queries
    ```

    - Add query chain into retrieval chain

    ```Python
    # retrieval_chain.py
    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]


    def get_retrieval_chain(vector_store_path, embeddings, template, model_name):
        vector_store = load_vector_store(
            index_dir_name=vector_store_path, embeddings=embeddings)
        retrieval = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )
        retrieval_chain = get_query_chain(
            template=template, model_name=model_name) | retrieval.map() | get_unique_union
        return retrieval_chain


    def load_vector_store(index_dir_name, embeddings):
        try:
            return FAISS.load_local(
                index_dir_name,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(e)
            return None
    ```

3. Generation: `app/crud/dating_crud.py`, `app/crud/rag_methods/rag_chain.py` and `app/prompt_templates/prompt_templates_enum.py`
    - Prompt Templates for Generation

    ```Python
    # prompt_templates_enum.py
    class PromptsEnum(Enum):
        ...
        answer_gen_template = """
            - You are a helpful assistant that answers questions about the context below.
            - You do not make up answers to questions that cannot be found in the context.
            - If you don't know the answer to a question, just say that you don't know. Don't try to make up an answer.
            - You will generate a list of activities and please follow the format:
            [
            {{
                "activityTitle": Get the name of the place,
                "activityLoc": Get the address of the place,
                "timeTotal": Generate your expected time about the place or just put 1 hour,
                "activityDescription": Generate a description of the place based on your understanding,
                "activityImage": Get url of the place
            }},
            ...
            ]
            - Make sure the list contains at least 5 activities
            - You have to answer in Korean.

            Answer the following question based on this context:

            {context}

            Question: {question}
        """
    ```

    - Creating RAG Chain for Retrieval and Augumented Generation

    ```Python
    # dating_crud.pu
    def dating_generation(request: DatingGenRequestDto) -> list[DatingGenResponseDto]:
        ...
        # RAG Chain
        print("Rag Chain")
        rag_chain = get_rag_chain(
            template=answer_gen_template,
            model_name="gpt-3.5-turbo",
            retrieval_chain=retrieval_chain,
        )
        ...
    ```

    - RAG Chain

    ```Python
    # rag_chain.py
    def get_rag_chain(template, retrieval_chain, model_name):
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model=model_name, temperature=0)
        final_rag_chain = (
            {"context": retrieval_chain,
            "question": itemgetter("question")}
            | prompt
            | llm
            | StrOutputParser()
        )
        return final_rag_chain
    ```

4. Response to Client: `app/schema/response_dto/dating_generation_response_dto.py`, `app/crud/dating_crud.py` and `app/crud/rag_methods/result_parsing.py`
    - Response Structure

    ```Python
    # dating_generation_response_dto.py
    class DatingGenResponseDto(BaseModel):
        activityTitle: str
        activityLocation: str
        timeTotal: str
        activityDescription: str
        activityImage: str
    ```

    - Generation and format check

    ```Python
    # dating_crud.py
    def dating_generation(request: DatingGenRequestDto) -> list[DatingGenResponseDto]:
        ...
        generated_answer = rag_chain.invoke({"question": question_query})
        generate_dating_responses = get_dto(generated_answer)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"FAISS index initialization took {elapsed_time:.4f} seconds")
        return generate_dating_responses
    ```

    - Format check for client

    ```Python
    # result_parsing.py
    def get_dto(result):
        activities_list = json.loads(result)
        dating_generation_responses = []
        for activity in activities_list:
            response_dto = DatingGenResponseDto(
                activityTitle=activity['activityTitle'],
                activityLocation=activity['activityLoc'],
                timeTotal=activity['timeTotal'],
                activityDescription=activity['activityDescription'],
                activityImage=activity['activityImage']
            )
            dating_generation_responses.append(
                response_dto
            )
        return dating_generation_responses
    ```

### Result

![Service Architecture](/static/project//project_result.png)
