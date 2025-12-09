from flask.cli import load_dotenv
from langchain_core.prompts import PromptTemplate
from output_parser import parser
from langchain_openai import AzureChatOpenAI
import json
import os

class LLM:
    
    def __init__(self, azure_deployment, openai_api_version, temperature):
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        self.model = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
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

    def generate_personalised_feedback(self, attempt_data):
        """
        Generate personalized, educational feedback for student's quest attempt
        """
        # Calculate basic statistics
        answers_list = attempt_data.get('answers', [])
        correct_count = sum(1 for ans in answers_list if ans.get('is_correct'))
        total_count = len(answers_list)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

        prompt = PromptTemplate(
            template="You are an educational AI tutor. Analyze this student's quiz attempt and provide detailed, "
                     "constructive, and encouraging feedback.\n\n"
                     "STUDENT PERFORMANCE SUMMARY:\n"
                     "- Total Questions: {total_questions}\n"
                     "- Correct Answers: {correct_answers}\n"
                     "- Accuracy: {accuracy}%\n\n"
                     "DETAILED ANSWERS:\n{attempt_data}\n\n"
                     "INSTRUCTIONS:\n"
                     "Provide feedback in the following JSON format (return ONLY valid JSON, no markdown, no extra text):\n"
                     "{{\n"
                     '    "strengths": [\n'
                     '        "Specific strength based on correct answers (e.g., Strong understanding of Python Lists)",\n'
                     '        "Another specific strength (e.g., Correctly applied concepts in scenario-based questions)"\n'
                     "    ],\n"
                     '    "weaknesses": [\n'
                     '        "Specific area needing improvement (e.g., Struggles with algorithm complexity analysis)",\n'
                     '        "Another weakness (e.g., Needs practice with code debugging)"\n'
                     "    ],\n"
                     '    "recommendations": "A detailed paragraph (3-5 sentences) with actionable study recommendations. '
                     'Be specific about topics to review, resources to use, and practice activities.",\n'
                     '    "question_feedback": {{\n'
                     '        "question_id_1": {{\n'
                     '            "feedback": "Specific feedback explaining why the answer was correct/incorrect",\n'
                     '            "concept_explanation": "Clear explanation of the underlying concept",\n'
                     '            "study_tip": "Actionable tip for improvement on this topic"\n'
                     "        }}\n"
                     "    }}\n"
                     "}}\n\n"
                     "IMPORTANT GUIDELINES:\n"
                     "1. Focus on Bloom's taxonomy levels where student struggled\n"
                     "2. Identify specific topics/concepts needing review\n"
                     "3. Use an encouraging, supportive tone - emphasize growth mindset\n"
                     "4. Provide actionable, concrete steps for improvement\n"
                     "5. Address common misconceptions if detected\n"
                     "6. For correct answers, reinforce why they were correct\n"
                     "7. For incorrect answers, explain the mistake and how to avoid it\n"
                     "8. Return ONLY the JSON object, no additional text before or after",
            input_variables=["total_questions", "correct_answers", "accuracy", "attempt_data"]
        )

        chain = prompt | self.model

        try:
            result = chain.invoke({
                "total_questions": total_count,
                "correct_answers": correct_count,
                "accuracy": f"{accuracy:.1f}",
                "attempt_data": json.dumps(answers_list, indent=2)
            })

            # Parse the response content as JSON
            content = result.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            feedback_json = json.loads(content)
            
            # Validate required keys
            required_keys = ['strengths', 'weaknesses', 'recommendations', 'question_feedback']
            if not all(key in feedback_json for key in required_keys):
                raise ValueError("Missing required feedback fields")
            
            print(f"[LLM Feedback] Generated successfully")
            return feedback_json
            
        except json.JSONDecodeError as e:
            print(f"[LLM Feedback Error] JSON parsing failed: {str(e)}")
            
            # Return fallback structure
            return {
                "strengths": [
                    f"Completed the quiz with {accuracy:.1f}% accuracy",
                    "Demonstrated effort in attempting all questions"
                ],
                "weaknesses": [
                    "Review the material covered in this quiz",
                    "Practice more questions to improve understanding"
                ],
                "recommendations": f"You scored {accuracy:.1f}% on this quiz. Focus on reviewing the topics where "
                                   f"you made mistakes. Consider re-reading the course materials and practicing "
                                   f"similar problems. Keep practicing - each attempt helps you improve!",
                "question_feedback": {}
            }
            
        except Exception as e:
            print(f"[LLM Feedback Error] Unexpected error: {str(e)}")
            
            # Return minimal fallback structure
            return {
                "strengths": ["Completed the quiz"],
                "weaknesses": ["Review the material"],
                "recommendations": "Keep practicing to improve your understanding.",
                "question_feedback": {}
            }
