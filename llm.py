from langchain_core.prompts import PromptTemplate
from output_parser import parser
from langchain_openai import AzureChatOpenAI


class LLM:
    def __init__(self, azure_deployment, openai_api_version, temperature):
        self.model = AzureChatOpenAI(
            openai_api_version=openai_api_version,
            azure_deployment=azure_deployment,
            temperature=temperature,
        )

    def generate_questions_and_answers(self, document_content, num_questions, difficulty):
        # test = "You should only respond in the following format in JSON: \n\n"
        #      "[ // List of questions\n"
        #      " { // Question 1\n"
        #      "  'number': 1, // Question number\n"
        #      "  'text': 'What is the capital of France?', // Question text\n"
        #      "  'answers': [ // List of answers\n"
        #      "    { // Answer 1\n"
        #      "      'text': 'Paris', // Answer text\n"
        #      "      'is_correct': true // Whether the answer is correct\n"
        #      "    },\n"
        #      "    { // Answer 2\n"
        #      "      'text': 'London', // Answer text\n"
        #      "      'is_correct': false // Whether the answer is correct\n"
        #      "    }\n"
        #      "  ]\n"
        #      " }\n"
        #      "]\n\n"
        prompt = PromptTemplate(
            template="You are a helpful learning assistant for students. Your goal is to facilitate their learning by "
                     "testing their understanding of the content from a lecture note. Based on the provided lecture "
                     "document, generate {num_questions} questions. Ensure that these questions are of {difficulty} "
                     "difficulty. A question should include a list of answers, and each answer has an indication "
                     "whether it is a correct answer.\n\n"
                     "Below is the content of the lecture document:\n\n{document_content}",
            input_variables=["num_questions", "difficulty", "document_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model | parser

        print(chain)

        result = chain.invoke({
            "num_questions": num_questions,
            "difficulty": difficulty,
            "document_content": document_content
        })

        print(result)

        return result




