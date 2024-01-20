from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import requests
import os
import streamlit as st


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    text = image_to_text(url)[0]['generated_text']
    return text


def generate_story(scenario):
    template = """
    you are a story teller;
    you can generate a short story based on a single narrative, the story should be no more than 20 words;
    CONTEXT: {scenario}
    STORY: 
    """

    prompt = PromptTemplate(template=template, input_variables=['scenario'])
    story_llm = LLMChain(llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="IMAGE TO AUDIO STORY")
    st.header("Turn image into story")
    upload_file = st.file_uploader("choose an image....", type='jpg')

    if upload_file is not None:
        print(upload_file)
        bytes_data = upload_file.getvalue()
        with open(upload_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(upload_file, caption="Uploaded image", use_column_width=True)
        scenario = img2text(upload_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")


# c = img2text(r""E:\projects\Image to audio\istockphoto-1457889029-612x612.jpg"")
# story = generate_story(c)
# text2speech(story)


if __name__ == '__main__':
    main()
