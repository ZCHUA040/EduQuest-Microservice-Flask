from langchain_core.prompts import PromptTemplate
from output_parser import parser
from langchain_openai import AzureChatOpenAI
import json


class LLM:
    def __init__(self, azure_deployment, openai_api_version, temperature):
        self.model = AzureChatOpenAI(
            openai_api_version=openai_api_version,
            azure_deployment=azure_deployment,
            temperature=temperature,
        )

    def generate_questions_and_answers(self, document_content, num_questions, difficulty):
        prompt = PromptTemplate(
            template="You are a helpful learning assistant for students. Your goal is to facilitate their learning by "
                     "testing their understanding of the content from a lecture note. Based on the provided lecture "
                     "document, generate {num_questions} questions. Ensure that these questions are of {difficulty} "
                     "difficulty. A question should include a list of 4 answers, and each answer has an indication "
                     "whether it is a correct answer and a reason to justify why this answer is correct or incorrect. "
                     "This is a multi select question and there can be more than one correct answer. "
                     "The possible answers does not have to be solely from the content of the document. You may also "
                     "generate other possible answers depending on the difficulty level. "
                     "{format_instructions} \n\n"
                     "Below is the content of the lecture document:\n\n{document_content}",
            input_variables=["num_questions", "difficulty", "document_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model | parser

        result = chain.invoke({
            "num_questions": num_questions,
            "difficulty": difficulty,
            "document_content": document_content
        })

        return result





