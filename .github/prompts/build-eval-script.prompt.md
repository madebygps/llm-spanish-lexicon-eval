---
mode: agent
---
# overview


You are an expert Python developer with an interest in understanding and leveraging your Python skills to experiment and evaluate open weight models and their understanding of Spanish. Our goal is to build an evaluation script with Python that will iterate over a list of words and use a specific prompt to prompt a list of models for their understanding of the words. We will then leverage another model to judge them and return all the results, additionally, a summary of how all the models performed. 

## task list


### Step 1: load files

 Look at the suite directory. Inside of here, we have models_list.txt with a list of models that we want to prompt. In this dot txt file, there are some lines that have a pound symbol. Those will indicate ones that I want in the list, but not actively do I want to evaluate.

Then take a look at prompts dot json. This file contains two prompts. At the moment, we're only going to be leveraging prompt a. Additionally, in this directory, we have a vocabulary underscore complete dot json and vocabulary underscore short dot json.

We only want to work with the short one for now. The first step is to load the words, load the list of models, and load the prompts. 

### Step 2: Prompt the models

We're going to be using open weight language models for this. They are all running locally with OLAMA and we're going to connect to them using the open AI package.

We then want to prompt each of the models in the list of models. We want to make sure that for every time we prompt it, we save the raw response. And we save these in an output directory and then inside of a directory with the name of the model.

Make sure to use some sort of progress bar package so we can visually see the progress of each model as we go. 

I would like the raw response to contain the word and then the real definition and then the response that the model gave us. And then I would like the last value for each item in here to have a space where a judge can provide either correct or incorrect. 

### Step 3: judge 

Next, what we want to do is iterate through the responses given to us by each of the models and use the open AI package to use GPT-5 and ask it to say if the definition was either correct or incorrect. That's all we want the judging to do.

Once the judge has provided a response, go back and add that response to the raw response of the model. 

Also make sure to use a progress bar package or the judge progress here. 


### Step 4: generate summary

Once all the judging has been done, I'd like you to create a summary.json with the name of the model and then the percentage they got correct. I'd also like you to print a table out. You can use the rich package for that. 

## Guidelines

You must always follow these guidelines. 

- Keep the code concise, efficient, and Pythonic. 
- Keep comments to a minimum only when absolutely necessary. 
- If you need to run any code, use the uv run main.py.
- If you need to add a package, use uv add
- Do not use the Python command. Do not use the PIP command, nothing. Only these. 