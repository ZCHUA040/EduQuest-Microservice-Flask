from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
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
    hint: Optional[str] = Field(description="A short hint to help the student.", default=None)
    question_type: str = Field(description="Question type: mcq, matching, categorising, latex_mcq.", default="mcq")
    structured_data: Dict[str, Any] = Field(description="Extra data for non-mcq question types.", default_factory=dict)
    answers: List[Answer] = Field(description="The list of answers for the question.")

    @validator('number', 'text', 'hint', 'question_type', 'structured_data', 'answers', pre=True, always=True)
    def validate_fields(cls, v, field):
        if field.name == 'number' and not isinstance(v, int):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'text' and not isinstance(v, str):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'hint' and v is not None and not isinstance(v, str):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'question_type' and not isinstance(v, str):
            raise ValueError(f"Invalid type for {field.name}: {v}")
        if field.name == 'structured_data' and not isinstance(v, dict):
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
