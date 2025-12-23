from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load language model
model = SentenceTransformer('all-MiniLM-L6-v2')

def check_hallucination(correct_answer, ai_answer):
    # Convert text to numbers
    emb1 = model.encode([correct_answer])
    emb2 = model.encode([ai_answer])

    # Compare similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]

    if similarity < 0.6:
        return "❌ Hallucination", similarity
    else:
        return "✅ Correct", similarity
    
# -------- TEST CASE --------
correct = "Isaac Newton discovered gravity"
ai_output = "Albert Einstein discovered gravity"

result, score = check_hallucination(correct, ai_output)

print("Result:", result)
print("Similarity Score:", score)
print("\n" + "="*60)
print("Testing with CSV file")
print("="*60 + "\n")

# Load CSV file
data = pd.read_csv('truthfulqa_sample.csv')

# Test each row
for idx, row in data.iterrows():
    question = row['question']
    correct_answer = row['correct_answer']
    ai_answer = row['ai_answer']
    
    result, score = check_hallucination(correct_answer, ai_answer)
    
    print(f"Question {idx + 1}: {question}")
    print(f"Correct Answer: {correct_answer}")
    print(f"AI Answer: {ai_answer}")
    print(f"Result: {result}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 60)
