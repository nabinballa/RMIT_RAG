import subprocess
import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import os

import nltk
nltk.download('wordnet')

# Load all training data CSVs and combine into one DataFrame
files = [
    'data/housing.csv',
    'data/myki.csv',
    'data/oshc_providers.csv',
    'data/work_and_money.csv',
    'data/emergency_services.csv'
]

# Load training data
try:
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError(f"File {file} is empty")
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    all_answers = list(combined_df['answer'])
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Load queries from test_queries.csv (with 'query' and 'type' columns)
try:
    queries_df = pd.read_csv('test.csv')
    if queries_df.empty:
        raise ValueError("No queries found in test.csv")
    queries = list(queries_df['query'])
    query_types = list(queries_df['type'])  # 'exact', 'inferred', 'irrelevant'
except FileNotFoundError:
    print("Error: test.csv not found")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Function to run query and capture output
def run_query(query, k=5):
    cmd = f"make a QUESTION=\"{query}\" K={k}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        with open('debug_output.txt', 'a') as f:
            f.write(f"Query: {query}\nCommand: {cmd}\nOutput: {output}\nStderr: {stderr}\n{'='*50}\n")
        return output
    except subprocess.TimeoutExpired:
        print(f"Error: Query '{query}' timed out")
        with open('debug_output.txt', 'a') as f:
            f.write(f"Query: {query}\nError: Timeout\n{'='*50}\n")
        return ''
    except Exception as e:
        print(f"Error running query '{query}': {e}")
        with open('debug_output.txt', 'a') as f:
            f.write(f"Query: {query}\nError: {str(e)}\n{'='*50}\n")
        return ''

# Compute BERTScore and ROUGE-L
def compute_scores(generated_answer, reference_answers):
    if not generated_answer or not reference_answers:
        return 0.0, 0.0
    
    # BERTScore
    bert_f1_scores = []
    for ref in reference_answers:
        P, R, F1 = bert_score([generated_answer], [ref], model_type='distilbert-base-uncased', lang='en', verbose=False)
        bert_f1_scores.append(F1.item())
    bert_f1 = max(bert_f1_scores) if bert_f1_scores else 0.0
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, generated_answer)['rougeL'].fmeasure for ref in reference_answers]
    rougeL_f1 = max(rouge_scores) if rouge_scores else 0.0
    
    return bert_f1, rougeL_f1

# Evaluate retrieval and answer
def evaluate_unsupervised(answer, query_type):
    fallback_phrases = ['don\'t know', 'no relevant information', 'not found', 'not enough information']
    is_fallback = any(phrase in answer.lower() for phrase in fallback_phrases)
    is_relevant = query_type != 'irrelevant'
    precision = 1.0 if (is_relevant and not is_fallback) or (not is_relevant and is_fallback) else 0.0
    
    # Compute scores for relevant queries
    bert_f1, rougeL_f1 = (0.0, 0.0)
    if is_relevant and not is_fallback:
        bert_f1, rougeL_f1 = compute_scores(answer, all_answers)
    
    # Fallback score for irrelevant queries (used as faithfulness)
    fallback_score = 1.0 if not is_relevant and is_fallback else 0.0 if not is_relevant else None
    faithfulness = fallback_score if not is_relevant else 0.0  # Faithfulness is fallback for irrelevant, 0 for relevant
    
    with open('debug_evaluation.txt', 'a') as f:
        f.write(f"Query Type: {query_type}\n")
        f.write(f"Is Relevant: {is_relevant}\n")
        f.write(f"Is Fallback: {is_fallback}\n")
        f.write(f"Precision: {precision}\n")
        if is_relevant:
            f.write(f"BERTScore F1: {bert_f1}\n")
            f.write(f"ROUGE-L F1: {rougeL_f1}\n")
        if fallback_score is not None:
            f.write(f"Fallback Score (Faithfulness): {fallback_score}\n")
        f.write(f"Faithfulness: {faithfulness}\n")
        f.write("\n")
    
    return {
        'precision': precision,
        'bert_f1': bert_f1 if is_relevant else None,
        'rougeL_f1': rougeL_f1 if is_relevant else None,
        'faithfulness': faithfulness,
        'fallback_score': fallback_score
    }

# Run evaluations
results = []
total_precision, total_bert_f1, total_rougeL_f1, total_faithfulness, total_fallback = 0, 0, 0, 0, 0
relevant_count = sum(1 for t in query_types if t != 'irrelevant')
irrelevant_count = len(query_types) - relevant_count

# Clear debug files
if os.path.exists('debug_output.txt'):
    os.remove('debug_output.txt')
if os.path.exists('debug_evaluation.txt'):
    os.remove('debug_evaluation.txt')

for i, (query, q_type) in enumerate(zip(queries, query_types), 1):
    print(f"Processing query {i}/{len(queries)}: {query} ({q_type})")
    answer = run_query(query)
    metrics = evaluate_unsupervised(answer, q_type)
    results.append({
        'query': query,
        'type': q_type,
        'answer': answer,
        **metrics
    })
    total_precision += metrics['precision']
    if metrics['bert_f1'] is not None:
        total_bert_f1 += metrics['bert_f1']
    if metrics['rougeL_f1'] is not None:
        total_rougeL_f1 += metrics['rougeL_f1']
    total_faithfulness += metrics['faithfulness']
    if metrics['fallback_score'] is not None:
        total_fallback += metrics['fallback_score']

# Generate report
avg_precision = total_precision / len(queries) if queries else 0.0
avg_bert_f1 = total_bert_f1 / relevant_count if relevant_count > 0 else 0.0
avg_rougeL_f1 = total_rougeL_f1 / relevant_count if relevant_count > 0 else 0.0
avg_faithfulness = total_faithfulness / len(queries) if queries else 0.0
avg_fallback = total_fallback / irrelevant_count if irrelevant_count > 0 else 0.0

with open('evaluation_report.txt', 'w') as f:
    f.write(f"Average Retrieval Precision: {avg_precision:.2f}\n")
    f.write(f"Average BERTScore F1 (Relevant Queries): {avg_bert_f1:.2f}\n")
    f.write(f"Average ROUGE-L F1 (Relevant Queries): {avg_rougeL_f1:.2f}\n")
    f.write(f"Average Faithfulness Score (Irrelevant Queries Only): {avg_faithfulness:.2f}\n")
    f.write(f"Average Fallback Accuracy (Irrelevant Queries): {avg_fallback:.2f}\n\n")
    for res in results:
        f.write(f"Query: {res['query']} ({res['type']})\n")
        f.write(f"Answer: {res['answer']}\n")
        f.write(f"Precision: {res['precision']:.2f}\n")
        if res['bert_f1'] is not None:
            f.write(f"BERTScore F1: {res['bert_f1']:.2f}\n")
        if res['rougeL_f1'] is not None:
            f.write(f"ROUGE-L F1: {res['rougeL_f1']:.2f}\n")
        f.write(f"Faithfulness: {res['faithfulness']:.2f}\n")
        if res['fallback_score'] is not None:
            f.write(f"Fallback Score: {res['fallback_score']:.2f}\n")
        f.write("\n")

print("Evaluation complete. Report saved to evaluation_report.txt")
print("Debug output saved to debug_output.txt")
print("Evaluation debug saved to debug_evaluation.txt")