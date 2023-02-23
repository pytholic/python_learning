# from prefect import flow


# @flow
# def my_favorite_function():
#     print("What is your favorite number?")
#     return 42


# print(my_favorite_function())

# import requests
# from prefect import flow


# @flow
# def call_api(url):
#     return requests.get(url).json()


# api_result = call_api("http://time.jsontest.com/")
# print(api_result)

# import requests
# from prefect import flow, task


# @task
# def call_api(url):
#     response = requests.get(url)
#     print(response.status_code)
#     return response.json()


# @flow
# def api_flow(url):
#     fact_json = call_api(url)
#     return fact_json


# print(api_flow("https://catfact.ninja/fact"))

import requests
from prefect import flow, task


@task
def call_api(url):
    response = requests.get(url)
    print(response.status_code)
    return response.json()


@task
def parse_fact(response):
    fact = response["fact"]
    print(fact)
    return fact


@flow
def api_flow(url):
    fact_json = call_api(url)
    fact_text = parse_fact(fact_json)
    return fact_text


api_flow("https://catfact.ninja/fact")
