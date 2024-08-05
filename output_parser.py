from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List
from langchain_core.output_parsers import JsonOutputParser


class Answer(BaseModel):
    text: str = Field(description="The answer text.")
    is_correct: bool = Field(description="Whether the answer is correct for the question.", default=False)
    reason: str = Field(description="The reason why the answer is correct or incorrect.")

    @validator('text', 'is_correct', 'reason', pre=True, always=True)
    def validate_fields(cls, v, field):
        if field.name == 'text' and not isinstance(v, str):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'is_correct' and not isinstance(v, bool):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'reason' and not isinstance(v, str):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        return v


class Question(BaseModel):
    number: int = Field(description="The question number. Starting from 1.")
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


parser = JsonOutputParser(pydantic_object=QuestionList)
