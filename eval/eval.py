import subprocess
import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import os
import time

import nltk
nltk.download('wordnet')

# Get the directory of eval.py and project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))  # eval folder
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_QUERIES_PATH = os.path.join(EVAL_DIR, 'test.csv') 

# Load all training data CSVs and combine into one DataFrame
files = [
    os.path.join(DATA_DIR, 'housing.csv'),
    os.path.join(DATA_DIR, 'myki.csv'),
    os.path.join(DATA_DIR, 'oshc_providers.csv'),
    os.path.join(DATA_DIR, 'work_and_money.csv'),
    os.path.join(DATA_DIR, 'emergency_services.csv')
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
    queries_df = pd.read_csv(TEST_QUERIES_PATH)
    if queries_df.empty:
        raise ValueError("No queries found")
    queries = list(queries_df['query'])
    query_types = list(queries_df['type'])  # 'exact', 'inferred', 'irrelevant'
except FileNotFoundError:
    print("Error: test.csv not found")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Function to run query and capture output
def run_query(query, k=1):
    cmd = f"make a QUESTION=\"{query}\" K={k}"
    start_time = time.perf_counter()  # to calculate execution time
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        response_time = time.perf_counter() - start_time  # Calculate response time
        return output, response_time
    except subprocess.TimeoutExpired:
        print(f"Error: Query '{query}' timed out")
        response_time = time.perf_counter() - start_time
        return '', response_time
    except Exception as e:
        print(f"Error running query '{query}': {e}")
        response_time = time.perf_counter() - start_time
        return '', response_time

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
    
    return {
        'precision': precision,
        'bert_f1': bert_f1 if is_relevant else None,
        'rougeL_f1': rougeL_f1 if is_relevant else None,
        'faithfulness': faithfulness,
        'fallback_score': fallback_score
    }

# Run evaluations
results = []
total_response_time = 0
total_precision, total_bert_f1, total_rougeL_f1, total_faithfulness, total_fallback = 0, 0, 0, 0, 0
relevant_count = sum(1 for t in query_types if t != 'irrelevant')
irrelevant_count = len(query_types) - relevant_count


for i, (query, q_type) in enumerate(zip(queries, query_types), 1):
    print(f"Processing query {i}/{len(queries)}: {query} ({q_type})")
    answer, response_time = run_query(query)
    metrics = evaluate_unsupervised(answer, q_type)
    results.append({
        'query': query,
        'type': q_type,
        'answer': answer,
        'response_time': response_time,
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
    total_response_time += response_time

pd.DataFrame(results).to_csv(os.path.join(EVAL_DIR, 'eval_results.csv'), index=False)

# Generate report
avg_precision = total_precision / len(queries) if queries else 0.0
avg_bert_f1 = total_bert_f1 / relevant_count if relevant_count > 0 else 0.0
avg_rougeL_f1 = total_rougeL_f1 / relevant_count if relevant_count > 0 else 0.0
avg_faithfulness = total_faithfulness / len(queries) if queries else 0.0
avg_fallback = total_fallback / irrelevant_count if irrelevant_count > 0 else 0.0
avg_response_time = total_response_time / len(queries) if queries else 0.0


with open(os.path.join(EVAL_DIR, 'evaluation_report.txt'), 'w') as f:
    f.write(f"Average Retrieval Precision: {avg_precision:.2f}\n")
    f.write(f"Average BERTScore F1 (Relevant Queries): {avg_bert_f1:.2f}\n")
    f.write(f"Average ROUGE-L F1 (Relevant Queries): {avg_rougeL_f1:.2f}\n")
    f.write(f"Average Faithfulness Score (Irrelevant Queries Only): {avg_faithfulness:.2f}\n")
    f.write(f"Average Fallback Accuracy (Irrelevant Queries): {avg_fallback:.2f}\n\n")
    f.write(f"Average Response Time (seconds): {avg_response_time:.3f}\n\n")
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
        f.write(f"Response Time (seconds): {res['response_time']:.3f}\n")
        f.write("\n")


print("Evaluation complete. Report saved to evaluation_report.txt")