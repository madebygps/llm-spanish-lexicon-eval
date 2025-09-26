---
mode: agent
---

# overview

We're going to build an evaluation script with Python. I want you to first load the JSON files, prompt.json, and vocabulary_complete.json that are in the suite directory. Then I would like you to iterate over every single word from the vocabulary file. Look at the list of models that we have in models_list.txt.

Use OLAMA for local models and then use OpenAI for the judge model. There is a .env file with an OpenAI API key available for you. 

Go through each model and prompt it, prompt A and prompt B from the prompts.json. For each model, I would like you to create a result. So it would be result_modelname.json containing responses that each model gives us. After that, I'd like you to call an OpenAI model to then judge.

I would like you to grab each model's response or results, and then ask the GBT model if the response is correct. And it needs to return yes or no. That's it. Store all of these judge responses as well, per each model. Then at the end, I need you to create results for each model, the percentage they got correct for prompt A, the percentage they got correct for prompt B, and that's it. 