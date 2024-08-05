from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import json


class Answer(BaseModel):
    text: str = Field(description="The answer text.")
    is_correct: bool = Field(description="Whether the answer is correct for the question.")


class Question(BaseModel):
    number: str = Field(description="The question number. Starting from 1.")
    text: str = Field(description="The question text.")
    answers: List[Answer] = Field(description="The list of answers for the question.")

    @validator('number', 'text', 'answers', pre=True, always=True)
    def validate_fields(cls, v, field):
        if field.name == 'number' and not isinstance(v, int):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'text' and not isinstance(v, str):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'answers' and not isinstance(v, list):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        return v

class QuestionList(BaseModel):
    questions: List[Question] = Field(description="The list of questions.")

    @validator('questions', pre=True, always=True)
    def validate_questions(cls, v):
        if not isinstance(v, list):
            raise ValueError(f"Invalid type for questions: {v}")
        return v


parser = PydanticOutputParser(pydantic_object=QuestionList)


if __name__ == '__main__':

    try:
        # Example usage
        question_list_data = {
            "questions": [
                {
                    "number": 1,
                    "text": "What is the capital of France?",
                    "answers": [
                        {"text": "Paris", "is_correct": True},
                        {"text": "London", "is_correct": False}
                    ]
                }
            ]
        }

        # Convert dictionary to JSON string
        question_list_json = json.dumps(question_list_data)

        # Parse JSON string
        question_list = parser.parse(question_list_json)

        # Output parsed data
        print(question_list)
    except OutputParserException as e:
        print(f"Failed to parse QuestionList: {e}")
