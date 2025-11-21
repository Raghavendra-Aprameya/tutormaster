"""
Answer Evaluator Node
Evaluates student answers against correct answers and provides scores out of 10.
Uses LLM to assess answer quality, completeness, and accuracy.
"""

import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()


class AnswerEvaluator:
    """
    Evaluates student answers against correct answers and provides scores.
    """
    
    def __init__(self):
        """Initialize the answer evaluator"""
        # Initialize Azure OpenAI
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=openai_endpoint,
            api_version="2024-12-01-preview",
            azure_deployment="gpt-4o-mini",
            temperature=0.3,  # Lower temperature for more consistent scoring
        )
    
    def evaluate_answer(self, question: str, correct_answer: str, student_answer: str) -> Dict:
        """
        Evaluate a student's answer against the correct answer.
        
        Args:
            question: The question that was asked
            correct_answer: The correct/reference answer
            student_answer: The student's answer to evaluate
            
        Returns:
            Dictionary containing score, feedback, and evaluation details
        """
        if not student_answer or not student_answer.strip():
            return {
                "score": 0,
                "max_score": 10,
                "feedback": "No answer provided.",
                "evaluation": {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0
                }
            }
        
        # Create evaluation prompt
        prompt_template = PromptTemplate(
            input_variables=["question", "correct_answer", "student_answer"],
            template="""You are an expert educator evaluating a student's answer for a History exam (Class 10).

Question: {question}

Correct Answer (Reference):
{correct_answer}

Student's Answer:
{student_answer}

TASK: Evaluate the student's answer and provide a score out of 10.

EVALUATION CRITERIA:
1. Accuracy (0-4 points): How correct are the facts and information provided?
2. Completeness (0-3 points): How well does the answer cover the key points from the correct answer?
3. Relevance (0-2 points): How relevant and on-topic is the answer?
4. Clarity (0-1 point): Is the answer well-structured and clear?

SCORING GUIDELINES:
- 9-10: Excellent - All key points covered accurately, well-structured, comprehensive
- 7-8: Good - Most key points covered, minor inaccuracies or omissions
- 5-6: Satisfactory - Some key points covered, but missing important information
- 3-4: Needs Improvement - Few key points covered, significant inaccuracies
- 1-2: Poor - Minimal relevant information, mostly incorrect
- 0: No answer or completely irrelevant

OUTPUT FORMAT (JSON only, no additional text):
{{
    "score": <integer 0-10>,
    "max_score": 10,
    "feedback": "<constructive feedback for the student>",
    "evaluation": {{
        "accuracy": <0-4>,
        "completeness": <0-3>,
        "relevance": <0-2>,
        "clarity": <0-1>
    }},
    "strengths": ["<strength1>", "<strength2>"],
    "improvements": ["<improvement1>", "<improvement2>"]
}}

Be fair but strict. Award points only for accurate information. Partial credit is acceptable for partially correct answers.

JSON Response:"""
        )
        
        prompt = prompt_template.format(
            question=question,
            correct_answer=correct_answer,
            student_answer=student_answer
        )
        
        try:
            # Get evaluation from LLM
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                evaluation = json.loads(json_str)
                
                # Validate and ensure score is within range
                if "score" in evaluation:
                    evaluation["score"] = max(0, min(10, int(evaluation.get("score", 0))))
                    evaluation["max_score"] = 10
                
                # Ensure all required fields exist
                if "feedback" not in evaluation:
                    evaluation["feedback"] = "Evaluation completed."
                if "evaluation" not in evaluation:
                    evaluation["evaluation"] = {
                        "accuracy": 0,
                        "completeness": 0,
                        "relevance": 0,
                        "clarity": 0
                    }
                if "strengths" not in evaluation:
                    evaluation["strengths"] = []
                if "improvements" not in evaluation:
                    evaluation["improvements"] = []
                
                return evaluation
            else:
                # Fallback if JSON parsing fails
                return {
                    "score": 0,
                    "max_score": 10,
                    "feedback": "Error: Could not parse evaluation response.",
                    "evaluation": {
                        "accuracy": 0,
                        "completeness": 0,
                        "relevance": 0,
                        "clarity": 0
                    },
                    "strengths": [],
                    "improvements": []
                }
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM: {str(e)}")
            print(f"Response: {response_text[:500]}")
            return {
                "score": 0,
                "max_score": 10,
                "feedback": "Error: Could not parse evaluation. Please try again.",
                "evaluation": {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "clarity": 0
                },
                "strengths": [],
                "improvements": []
            }
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {
                "score": 0,
                "max_score": 10,
                "feedback": f"Error during evaluation: {str(e)}",
                "evaluation": {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "clarity": 0
                },
                "strengths": [],
                "improvements": []
            }
    
    def evaluate_multiple_answers(self, evaluations: list) -> Dict:
        """
        Evaluate multiple student answers.
        
        Args:
            evaluations: List of dictionaries, each containing:
                - question: str
                - answer: str (correct answer)
                - student_answer: str
        
        Returns:
            Dictionary with overall results and individual evaluations
        """
        results = []
        total_score = 0
        max_total_score = 0
        
        for eval_data in evaluations:
            question = eval_data.get("question", "")
            correct_answer = eval_data.get("answer", "")
            student_answer = eval_data.get("student_answer", "")
            
            evaluation = self.evaluate_answer(question, correct_answer, student_answer)
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "student_answer": student_answer,
                "evaluation": evaluation
            })
            
            total_score += evaluation.get("score", 0)
            max_total_score += evaluation.get("max_score", 10)
        
        # Calculate percentage
        percentage = (total_score / max_total_score * 100) if max_total_score > 0 else 0
        
        return {
            "total_score": total_score,
            "max_total_score": max_total_score,
            "percentage": round(percentage, 2),
            "total_questions": len(results),
            "evaluations": results
        }


def main():
    """Test the answer evaluator"""
    evaluator = AnswerEvaluator()
    
    # Test evaluation
    question = "What were the key features of the Hindu World Order between 1000 B.C.E. and 700 C.E.?"
    correct_answer = "The Hindu World Order during this period was characterized by a complex caste system, the emergence of powerful kingdoms, and the establishment of religious traditions that shaped Indian society. The integration of local customs with Vedic practices also played a significant role in this evolution."
    student_answer = "The Hindu World Order had a caste system and kingdoms."
    
    result = evaluator.evaluate_answer(question, correct_answer, student_answer)
    
    print("=" * 80)
    print("ANSWER EVALUATION TEST")
    print("=" * 80)
    print(f"\nQuestion: {question}")
    print(f"\nCorrect Answer: {correct_answer}")
    print(f"\nStudent Answer: {student_answer}")
    print(f"\nScore: {result['score']}/10")
    print(f"\nFeedback: {result['feedback']}")
    print(f"\nEvaluation Breakdown:")
    print(f"  Accuracy: {result['evaluation']['accuracy']}/4")
    print(f"  Completeness: {result['evaluation']['completeness']}/3")
    print(f"  Relevance: {result['evaluation']['relevance']}/2")
    print(f"  Clarity: {result['evaluation']['clarity']}/1")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

