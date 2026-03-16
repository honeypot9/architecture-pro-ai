#!/usr/bin/env python3

import os
import json
import csv
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from index import SimpleRAGEngine, SafetyConfig

class RAGEvaluator:
    def __init__(self, questions_file: str = "golden_questions.txt"):
        self.questions = self._load_questions(questions_file)
        self.rag_engine = SimpleRAGEngine(SafetyConfig())
        self.results = []

    def _load_questions(self, filename: str) -> List[Dict[str, Any]]:
        """Load questions with expected answers"""
        questions = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                current_question = {}
                for line in f:
                    line = line.strip()
                    if line.startswith("Q:"):
                        if current_question:
                            questions.append(current_question)
                        current_question = {"question": line[2:].strip(), "expected_answer": ""}
                    elif line.startswith("A:") and current_question:
                        current_question["expected_answer"] = line[2:].strip()
                if current_question:
                    questions.append(current_question)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, using default questions")
            questions = [

            ]
        return questions

    def _evaluate_response(self, question: str, response_data: Dict[str, Any], expected: str) -> Dict[str, Any]:
        """Evaluate response against expected result"""
        has_answer = response_data.get('has_answer', False)
        answer = response_data.get('answer', '')
        answer_length = len(answer)
        num_sources = len(response_data.get('sources', []))

        # Determine if response is correct based on expectation
        if expected in answer:
            is_correct = has_answer and answer_length > 30
        else:  # expected "not_found"
            is_correct = not has_answer or answer_length < 100

        return {
            'question': question,
            'expected': expected,
            'has_answer': has_answer,
            'answer_length': answer_length,
            'num_sources': num_sources,
            'is_correct': is_correct,
            'response_preview': response_data.get('answer', '')[:100] + '...' if response_data.get('answer') else 'No answer',
            'timestamp': datetime.now().isoformat()
        }

    async def run_evaluation(self):
        """Run evaluation on all questions"""
        print("Starting RAG system evaluation...")

        for i, qa in enumerate(self.questions, 1):
            print(f"Testing {i}/{len(self.questions)}: {qa['question']}")

            try:
                response = await self.rag_engine.create_response(qa['question'])
                evaluation = self._evaluate_response(
                    qa['question'],
                    response,
                    qa['expected_answer']
                )
                self.results.append(evaluation)

                status = "✓" if evaluation['is_correct'] else "✗"
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results.append({
                    'question': qa['question'],
                    'expected': qa['expected_answer'],
                    'has_answer': False,
                    'answer_length': 0,
                    'num_sources': 0,
                    'is_correct': False,
                    'response_preview': f'Error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                })

    def generate_report(self):
        """Generate evaluation report"""
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        accuracy = correct / total if total > 0 else 0

        print(f"\n{'='*50}")
        print(f"EVALUATION REPORT")
        print(f"{'='*50}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")

        # Detailed breakdown
        found_questions = [r for r in self.results if r['expected'] == 'found']
        not_found_questions = [r for r in self.results if r['expected'] == 'not_found']

        if found_questions:
            found_correct = sum(1 for r in found_questions if r['is_correct'])
            print(f"\nQuestions that should be answered:")
            print(f"  Correct: {found_correct}/{len(found_questions)} ({found_correct/len(found_questions):.2%})")

        if not_found_questions:
            not_found_correct = sum(1 for r in not_found_questions if r['is_correct'])
            print(f"Questions that should NOT be answered:")
            print(f"  Correct: {not_found_correct}/{len(not_found_questions)} ({not_found_correct/len(not_found_questions):.2%})")

        # Save detailed results
        with open('evaluation_results.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['question', 'expected', 'has_answer', 'answer_length', 'num_sources', 'is_correct', 'response_preview', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)

        print(f"\nDetailed results saved to: evaluation_results.csv")
        print(f"Interaction logs saved to: evaluation_logs.jsonl")

async def main():
    evaluator = RAGEvaluator()
    await evaluator.run_evaluation()
    evaluator.generate_report()

if __name__ == "__main__":
    asyncio.run(main())
