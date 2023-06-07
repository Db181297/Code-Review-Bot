import os
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, OpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import SequentialChain
from langchain.memory import ConversationStringBufferMemory
import openai


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def review_code(program, language):
    "function to call OpenAI and get the review using sequential chain"

    # chain for reviewing the code
    review_system_template = """You are a code review assistant tasked with judging the quality and readability of a colleague's script.
                            Your role is to explain what each section of the code does, identify any potential issues, and suggest improvements
                            based on industry best practices. As you work through the code, provide clear explanations and bullet-pointed 
                            recommendations for how to optimize it. Imagine that your colleague is new to programming and may need extra 
                            guidance to understand your feedback.Please note that your role is to provide guidance and recommendations only; 
                            you are not supposed to make changes to the code yourself."""
    review_system_message_prompt = SystemMessagePromptTemplate.from_template(
        review_system_template)

    review_human_template = "Please review the following code: \n {code_snippet}"
    review_human_message_prompt = HumanMessagePromptTemplate.from_template(
        review_human_template)

    review_chat_prompt = ChatPromptTemplate.from_messages(
        [review_system_message_prompt, review_human_message_prompt])
    review_memory = ConversationStringBufferMemory(memory_key="chat_memories")
    review_model = ChatOpenAI(streaming=True,
                              callbacks=[StreamingStdOutCallbackHandler()],
                              temperature=0)
    review_chain = LLMChain(llm=review_model,
                            prompt=review_chat_prompt,
                            output_key="bullet_points",
                            memory=review_memory)

    # chain for updating the code
    update_system_template = f"""As an assistant tasked with refactoring a codebase written in {language}, your goal is to optimize its efficiency and functionality
                            while maintaining readability. You will carefully read the provided explanation of the code and review all 
                            suggested changes one by one, after considering the rationale of each recommendation, implement them to refactor 
                            the code only if necessary. Finally after reviewing all the recommendations and refactoring return the updated 
                            version. Please note that it is essential to keep the names of methods, functions, classes, constants and variables 
                            used in the code unchanged.\n"""
    update_system_message_prompt = SystemMessagePromptTemplate.from_template(
        update_system_template)

    update_human_template = "{bullet_points}"
    update_human_message_prompt = HumanMessagePromptTemplate.from_template(
        update_human_template)

    update_chat_prompt = ChatPromptTemplate.from_messages(
        [update_system_message_prompt, update_human_message_prompt])
    update_model = ChatOpenAI(streaming=True,
                              callbacks=[StreamingStdOutCallbackHandler()],
                              temperature=0.5)
    update_chain = LLMChain(llm=update_model,
                            prompt=update_chat_prompt,
                            output_key="updated_code")

    # chain for adding comments to the code
    comment_system_template = """You are a helpful assistant that adds inline comments to source code to make it easier to understand.
                            Your comments should follow a clear and consistent style and provide additional context or explanations 
                            for readers. For example, you might add a comment to explain the purpose of a particular function or method, 
                            clarify complex logic, or highlight potential issues or bugs."""
    comment_system_message_prompt = SystemMessagePromptTemplate.from_template(
        comment_system_template)

    comment_human_template = "Please add only inline comments to this code: \n {updated_code}"
    comment_human_message_prompt = HumanMessagePromptTemplate.from_template(
        comment_human_template)

    comment_chat_prompt = ChatPromptTemplate.from_messages(
        [comment_system_message_prompt, comment_human_message_prompt])
    comment_model = ChatOpenAI(streaming=True,
                               callbacks=[StreamingStdOutCallbackHandler()],
                               temperature=0.9)
    comment_chain = LLMChain(llm=comment_model,
                             prompt=comment_chat_prompt,
                             output_key="code_w_comments")

    # run all the 3 chains in sequential order
    overall_chain = SequentialChain(chains=[review_chain, update_chain, comment_chain],
                                    input_variables=["code_snippet"],
                                    output_variables=[
                                        "bullet_points", "code_w_comments"],
                                    verbose=True,)
    result = overall_chain({"code_snippet": program})

    return result


def convert_code_to_python(code, language):
    # Define the prompt for code conversion
    prompt = f"Translate the following {language} code to Python:\n\n{code}"

    # Generate code using OpenAI Codex API
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.0,
        n=1,
        stop=None,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract and return the generated Python code
    converted_code = response.choices[0].text
    prompt = f"{converted_code}\n\n\"\"\"\nHere's what the above code does:\n"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
    )

    if response.choices:
        explanation = response.choices[0].text
    else:
        explanation = "No response generated."
    return explanation

# Define the API endpoint for code conversion


@app.route('/api/convert', methods=['POST'])
def convert_code():
    data = request.get_json()
    code = data.get('code')
    language = data.get('language')

    # Perform code conversion logic here
    converted_code = convert_code_to_python(code, language)
    result = review_code(code, language)

    return jsonify({'converted_code': converted_code, 'bullet_points': result["bullet_points"], 'code_w_comments': result['code_w_comments']})


# Define the API endpoint for code review
# @app.route('/api/review', methods=['POST'])
# def get_code():
#     data = request.get_json()
#     code = data.get('code')
#     language = data.get('language')

#     # Perform code review logic here
#     result = review_code(code, language)

#     return jsonify({'bullet_points': result["bullet_points"], 'code_w_comments': result['code_w_comments']})


if __name__ == '__main__':
    # get the OpenAI api key and set it as environment variable
    OPENAI_API_KEY = "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"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    app.run(host='0.0.0.0', port=5000, debug=True)
