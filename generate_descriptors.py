import os

import json
import pandas as pd
from tqdm import tqdm
import itertools

from descriptor_strings import stringtolist

import api_secrets


# def generate_prompt(category_name: str):
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
# A: There are several useful visual features to tell there is a lemur in a photo:
# - four-limbed primate
# - black, grey, white, brown, or red-brown
# - wet and hairless nose with curved nostrils
# - long tail
# - large eyes
# - furry bodies
# - clawed hands and feet

# Q: What are useful visual features for distinguishing a television in a photo?
# A: There are several useful visual features to tell there is a television in a photo:
# - electronic device
# - black or grey
# - a large, rectangular screen
# - a stand or mount to support the screen
# - one or more speakers
# - a power cord
# - input ports for connecting to other devices
# - a remote control

# Q: What are useful features for distinguishing a {category_name} in a photo?
# A: There are several useful visual features to tell there is a {category_name} in a photo:
# -
# """

def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return [
        {"role": "system", "content": f"""Q: What are useful visual features for distinguishing a lemur in a photo?
        A: There are several useful visual features to tell there is a lemur in a photo:
        - four-limbed primate
        - black, grey, white, brown, or red-brown
        - wet and hairless nose with curved nostrils
        - long tail
        - large eyes
        - furry bodies
        - clawed hands and feet

        Q: What are useful visual features for distinguishing a television in a photo?
        A: There are several useful visual features to tell there is a television in a photo:
        - electronic device
        - black or grey
        - a large, rectangular screen
        - a stand or mount to support the screen
        - one or more speakers
        - a power cord
        - input ports for connecting to other devices
        - a remote control \n \n Please directly output answers."""},
        {"role": "user", "content": f"""Q: What are useful features for distinguishing a {category_name} in a photo?
        A: There are several useful visual features to tell there is a {category_name} in a photo:
        -
        """}
    ]

def generate_region_prompt(category_name: str, region_name: str = 'Africa'):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return [
        {"role": "system", "content": f"""Q: What are useful features for distinguishing a bathtub in a photo that I took in Japan?
        A: There are several useful visual features to tell there is a bathtub in a photo that I took in Japan:
        - short in length and deep 
        - square shape 
        - wooden, plastic, or steel material 
        - white or brown color 
        - benches on side 
        - next to shower """},
        {"role": "user", "content": f"""Q: What are useful features for distinguishing a {category_name} in a photo that I took in {region_name}?
        A: There are several useful visual features to tell there is/are {category_name} in a photo that I took in {region_name}:
        -
        """}
    ]

def generate_region_prompt_gpt3(category_name: str, region_name: str = 'Africa'):
    return f"""Q: What are useful features for distinguishing a bathtub in a photo that I took in Japan?
        A: There are several useful visual features to tell there is a bathtub in a photo that I took in Japan:
        - short in length and deep 
        - square shape 
        - wooden, plastic, or steel material 
        - white or brown color 
        - benches on side 
        - next to shower 
        
        Q: What are useful features for distinguishing a {category_name} in a photo that I took in {region_name}?
        A: There are several useful visual features to tell there is/are {category_name} in a photo that I took in {region_name}:
        -
        """

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}
    
    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    # prompts = [generate_region_prompt(category.replace('_', ' ')) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)


def obtain_geode_descriptors_and_save_gpt3(filename, class_list, model_name='gpt-3.5-turbo-instruct', region_name=''):
    
    import openai
    openai.api_key = api_secrets.openai_api_key
    openai.organization = api_secrets.openai_org

    responses = {}
    descriptors = {}
    
    prompts = [generate_region_prompt_gpt3(category.replace('_', ' '), region_name) for category in class_list]
    # print(prompts)
    # prompts = [generate_region_prompt(category.replace('_', ' ')) for category in class_list]
    
    for category in class_list:
        prompt = generate_region_prompt_gpt3(category.replace('_', ' '), region_name)
        print(prompt)
        response_text = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=0.,
            max_tokens=100,
        ).choices[0].text
        descriptors[category] = stringtolist(response_text)

    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    # responses = [openai.Completion.create(model=model_name,
    #                                         prompt=prompt_partition,
    #                                         temperature=0.,
    #                                         max_tokens=100,
    #                                         ) for prompt_partition in partition(prompts, 20)]
    # response_texts = [r["text"] for resp in responses for r in resp['choices']]
    # descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    # descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    print(descriptors)
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)
    

def obtain_geode_descriptors_and_save(filename, class_list, model_name='gpt-3.5-turbo-0125', region_name=''):

    from openai import OpenAI
    os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=api_secrets.openai_org
    )

    responses = []
    descriptors = {}
    descriptors_list = []
    
    if not region_name:
        prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    else:
        prompts = [generate_region_prompt(category.replace('_', ' '), region_name) for category in class_list]
    
    # for category in class_list:
    #     prompt = generate_region_prompt(category.replace('_', ' '), region_name)
    #     response_text = client.chat.completions.create(
    #         model="gpt-4",
    #         messages=prompt
    #     ).choices[0].message.content
    #     descriptors_list.append(stringtolist(response_text))
    #     print(response_text)
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [client.chat.completions.create(
            model=model_name,
            messages=prompt
        )  for prompt_partition in partition(prompts, 20) for prompt in prompt_partition]
    response_texts = [r.choices[0].message.content for r in responses]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # check if there is empty list as values or values as single element list in descriptors
    # if so, query the model again for the missing descriptors
    while any([not value or len(value) == 1 for value in descriptors.values()]):
        key = next(key for key, value in descriptors.items() if not value or len(value) == 1)
        print(key)
        prompt = generate_region_prompt(key.replace('_', ' '), region_name)
        response_text = client.chat.completions.create(
            model=model_name,
            messages=prompt
        ).choices[0].message.content
        descriptors[key] = stringtolist(response_text)
        print(response_text)

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp, indent=4)


df = pd.read_csv('/local2/data/xuanming/geode/index.csv', index_col=False)
class_list = list(df['object'].unique())  # class list in Geo-DE
# obtain_geode_descriptors_and_save_gpt3('descriptors_geode_westasia', class_list, model_name='gpt-3.5-turbo-instruct', region_name='West Asia')

#### country LLM prompting
orig_root_path = "/local2/data/xuanming/"
africa_index = pd.read_csv(orig_root_path + "geode_africa/index.csv")
eastasia_index = pd.read_csv(orig_root_path + "geode_eastasia/index.csv")
europe_index = pd.read_csv(orig_root_path + "geode_europe/index.csv")
southeastasia_index = pd.read_csv(orig_root_path + "geode_southeastasia/index.csv")
westasia_index = pd.read_csv(orig_root_path + "geode_westasia/index.csv")
america_index = pd.read_csv(orig_root_path + "geode_americas/index.csv")

africa_country = africa_index['ip_country'].unique().tolist()
eastasia_country = eastasia_index['ip_country'].unique().tolist()
europe_country = europe_index['ip_country'].unique().tolist()
southeastasia_country = southeastasia_index['ip_country'].unique().tolist()
westasia_country = westasia_index['ip_country'].unique().tolist()
america_country = america_index['ip_country'].unique().tolist()

root_path = "/local2/data/xuanming/geode_country/"
country_dict = {
    "africa": africa_country,
    "eastasia": eastasia_country,
    "europe": europe_country,
    "southeastasia": southeastasia_country,
    "westasia": westasia_country,
    "americas": america_country
}

for region in tqdm(country_dict):
    print(region)
    country_list = country_dict[region]
    for country in country_list:
        if ' ' in country:
            country_name = country.replace(' ', '_')
        else:
            country_name = country
        obtain_geode_descriptors_and_save_gpt3('./descriptors_country/' + f'descriptors_geode_{region}_{country_name.lower()}', class_list, model_name='gpt-3.5-turbo-instruct', region_name=country)