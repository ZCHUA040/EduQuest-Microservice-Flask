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
                     "generate other possible answers depending on the difficulty level.\n\n"
                     "ADDITIONAL REQUIREMENTS:\n"
                     "1) Avoid definition-only questions (max 20% if difficulty is Easy).\n"
                     "2) Ensure coverage across different topics/sections of the document; do not cluster on one topic.\n"
                     "3) Use plausible distractors based on common misconceptions.\n"
                     "4) For Medium/Hard, prioritize application and scenario-based questions.\n"
                     "5) Include at least 1 question that requires reasoning across multiple concepts.\n"
                     "6) Keep wording concise and unambiguous.\n"
                     "7) Keep answer options parallel in length and style.\n"
                     "8) Target distribution (approx.):\n"
                     "- Easy: Remember 30%, Understand 30%, Apply 20%, Analyze 10%, Evaluate 5%, Create 5%\n"
                     "- Medium: Remember 15%, Understand 25%, Apply 25%, Analyze 20%, Evaluate 10%, Create 5%\n"
                     "- Hard: Remember 10%, Understand 15%, Apply 25%, Analyze 25%, Evaluate 15%, Create 10%\n\n"
                     "QUESTION TYPE TAGGING:\n"
                     "Set question_type to one of: mcq, matching, categorising, latex_mcq.\n"
                     "Use latex_mcq for calculation-based questions; format math using LaTeX.\n"
                     "If question_type is matching or categorising, include a structured_data object:\n"
                     "- matching: {{\"pairs\": [{{\"left\": \"...\", \"right\": \"...\"}}]}}\n"
                     "- categorising: {{\"categories\": [{{\"name\": \"...\", \"items\": [\"...\"]}}]}}\n"
                     "For every question, include a short hint in a field named \"hint\".\n"
                     "Always include the standard 4-answer list for compatibility.\n"
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

        try:
            questions = result.get("questions", []) if isinstance(result, dict) else []
            for question in questions:
                hint = question.get("hint")
                if hint:
                    continue
                question_type = question.get("question_type", "mcq")
                if question_type == "matching":
                    question["hint"] = "Match each pair based on the core definitions."
                elif question_type == "categorising":
                    question["hint"] = "Group each item by the category definitions."
                elif question_type == "latex_mcq":
                    question["hint"] = "Set up the calculation carefully and check units."
                else:
                    question["hint"] = "Recall the key concept and eliminate distractors."
        except Exception:
            pass

        return result

    def generate_personalised_feedback(self, attempt_data):
        """
        Generate personalised, educational feedback for student's quest attempt
        """
        # Calculate basic statistics
        answers_list = attempt_data.get('answers', [])
        questions_map = {}
        for ans in answers_list:
            question_id = ans.get('question_id')
            if question_id is None:
                continue
            if question_id not in questions_map:
                questions_map[question_id] = {
                    "total_correct": 0,
                    "selected_correct": 0
                }
            answer_is_correct = ans.get('answer_is_correct')
            if answer_is_correct is None:
                answer_is_correct = ans.get('is_correct') and not ans.get('is_selected') is False
            if answer_is_correct:
                questions_map[question_id]["total_correct"] += 1
                if ans.get('is_selected'):
                    questions_map[question_id]["selected_correct"] += 1

        total_count = len(questions_map) or 0
        correct_count = sum(
            1 for stats in questions_map.values()
            if stats["total_correct"] > 0 and stats["selected_correct"] == stats["total_correct"]
        )
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

        prompt = PromptTemplate(
            template="You are an educational tutor. Analyze this student's quiz attempt and provide detailed, "
             "constructive, and encouraging feedback.\n\n"
             "STUDENT PERFORMANCE SUMMARY:\n"
             "- Total Questions: {total_questions}\n"
             "- Correct Answers: {correct_answers}\n"
             "- Accuracy: {accuracy}%\n\n"
             "DETAILED ANSWERS:\n{attempt_data}\n\n"
             "INSTRUCTIONS:\n"
             "Provide feedback in the following JSON format (return ONLY valid JSON, no markdown, no extra text):\n"
             "{{\n"
             '  "quest_summary": {{\n'
             '    "overall_bloom_rating": 1,\n'
             '    "overall_bloom_level": "Remember",\n'
             '    "summary": "2-3 sentence summary of performance across the quest."\n'
             "  }},\n"
             '  "subtopic_feedback": [\n'
             "    {{\n"
             '      "subtopic": "Subtopic name",\n'
             '      "bloom_rating": 2,\n'
             '      "bloom_level": "Understand",\n'
             '      "evidence": "Short evidence grounded in the student answers.",\n'
             '      "improvement_focus": "One sentence on what to improve in this subtopic."\n'
             "    }}\n"
             "  ],\n"
             '  "study_tips": [\n'
             '    "Practical study tip 1",\n'
             '    "Practical study tip 2"\n'
             "  ]\n"
             "}}\n\n"
             "BLOOM SCALE (STRICT 1-6)\n"
             "1 = Remember\n"
             "2 = Understand\n"
             "3 = Apply\n"
             "4 = Analyse\n"
             "5 = Evaluate\n"
             "6 = Create\n\n"
             "IMPORTANT GUIDELINES:\n"
             "1. Use ONLY the bloom levels listed and map them strictly to the 1-6 ratings\n"
             "2. Infer subtopics by grouping related questions; use concise subtopic names\n"
             "3. Provide 3-8 subtopic entries depending on coverage\n"
             "4. The quest summary should be 2-3 sentences and match the overall bloom rating\n"
             "5. Provide 3-6 study tips as a list, focused on the weakest subtopics\n"
             "6. Use an encouraging, supportive tone - emphasize growth mindset\n"
             "7. Return ONLY the JSON object, no additional text before or after",
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
            print("[LLM Raw]", content)
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            
            feedback_json = json.loads(content)
            
            # Validate required keys
            required_keys = ['quest_summary', 'subtopic_feedback', 'study_tips']
            if not all(key in feedback_json for key in required_keys):
                raise ValueError("Missing required feedback fields")

            print(f"[LLM Feedback] Generated successfully")
            return feedback_json

        except json.JSONDecodeError as e:
            print(f"[LLM Feedback Error] JSON parsing failed: {str(e)}")

            # Return fallback structure
            return {
                "quest_summary": {
                    "overall_bloom_rating": 2,
                    "overall_bloom_level": "Understand",
                    "summary": f"You completed the quest with {accuracy:.1f}% accuracy. Review the topics you missed and "
                               "practice applying them in new contexts for better mastery."
                },
                "subtopic_feedback": [],
                "study_tips": [
                    "Review the lesson notes for the questions you missed.",
                    "Redo similar practice questions to reinforce understanding.",
                    "Explain key concepts in your own words to check understanding."
                ]
            }

        except Exception as e:
            print(f"[LLM Feedback Error] Unexpected error: {str(e)}")

            # Return minimal fallback structure
            return {
                "quest_summary": {
                    "overall_bloom_rating": 1,
                    "overall_bloom_level": "Remember",
                    "summary": "Completed the quest. Review the materials and keep practicing."
                },
                "subtopic_feedback": [],
                "study_tips": ["Keep practicing to improve your understanding."]
            }

    def generate_bonus_game(self, document_content, game_type):
        if game_type == "matching":
            prompt = PromptTemplate(
                template="You are a learning assistant. Create a matching pairs mini-game based on the document.\n"
                         "Return ONLY valid JSON, no markdown.\n\n"
                         "FORMAT:\n"
                         "{{\n"
                         '  "game_type": "matching",\n'
                         '  "prompt": "...",\n'
                         '  "pairs": [\n'
                         '    {{"left": "...", "right": "..."}},\n'
                         '    {{"left": "...", "right": "..."}},\n'
                         '    {{"left": "...", "right": "..."}},\n'
                         '    {{"left": "...", "right": "..."}}\n'
                         '  ],\n'
                         '  "hint": "..."\n'
                         "}}\n\n"
                         "RULES:\n"
                         "- Generate 4 pairs.\n"
                         "- Keep text concise (<= 8 words each).\n"
                         "- Pairs must be clearly matched from the document.\n"
                         "- Avoid obscure or minor details.\n\n"
                         "DOCUMENT:\n{document_content}",
                input_variables=["document_content"]
            )
        else:
            prompt = PromptTemplate(
                template="You are a learning assistant. Create an ordering sequence mini-game based on the document.\n"
                         "Return ONLY valid JSON, no markdown.\n\n"
                         "FORMAT:\n"
                         "{{\n"
                         '  "game_type": "ordering",\n'
                         '  "prompt": "...",\n'
                         '  "items": ["...", "...", "...", "..."],\n'
                         '  "answer_order": [0, 1, 2, 3],\n'
                         '  "hint": "..."\n'
                         "}}\n\n"
                         "RULES:\n"
                         "- Generate 4 items in correct order in the items list.\n"
                         "- answer_order must be the correct index order (0..3).\n"
                         "- Use a process or sequence from the document.\n"
                         "- Keep items concise (<= 8 words each).\n\n"
                         "DOCUMENT:\n{document_content}",
                input_variables=["document_content"]
            )

        chain = prompt | self.model
        result = chain.invoke({
            "document_content": document_content
        })

        content = result.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()

        return json.loads(content)
