#!/usr/bin/env python3
"""
Simple pipeline for testing BIRD-CRITIC benchmark with LLM.
Supports iterative SQL debugging with execution feedback.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
import psycopg2
import requests


# Configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5-coder:32b"
MAX_ITERATIONS = 5


def load_dataset(dataset_path: str, sample_count: Optional[int] = None) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if sample_count:
        data = data[:sample_count]
    
    print(f"Loaded {len(data)} samples from {dataset_path}")
    return data


def connect_to_db(db_name: str, host: str = "localhost", port: int = 5432, 
                  user: str = "root", password: str = "123123") -> psycopg2.extensions.connection:
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database {db_name}: {e}")
        return None


def execute_sql(conn, sql: str) -> Dict:
    """Execute SQL query and return results or error."""
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        
        # Try to fetch results for SELECT queries
        try:
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            conn.commit()
            return {
                "success": True,
                "results": results,
                "columns": column_names,
                "rowcount": cursor.rowcount
            }
        except psycopg2.ProgrammingError:
            # No results to fetch (INSERT, UPDATE, DELETE, etc.)
            conn.commit()
            return {
                "success": True,
                "results": None,
                "rowcount": cursor.rowcount,
                "message": f"Query executed successfully. Rows affected: {cursor.rowcount}"
            }
    except Exception as e:
        conn.rollback()
        return {
            "success": False,
            "error": str(e)
        }


def call_llm(messages: List[Dict], model: str = DEFAULT_MODEL, 
             base_url: str = OLLAMA_BASE_URL) -> Optional[str]:
    """Call LLM API (OpenAI-compatible)."""
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def build_initial_prompt(sample: Dict) -> str:
    """Build initial prompt for the LLM."""
    prompt = f"""You are a SQL debugging expert. A user has reported an issue with their SQL query.

Database: {sample['db_id']}
User Query: {sample['query']}
Buggy SQL: {sample['issue_sql']}

Your task is to:
1. Analyze the buggy SQL query
2. Identify the issue
3. Provide a corrected SQL query
4. Test your solution iteratively

When you provide a SQL query to test, use this exact format:
```sql
YOUR SQL QUERY HERE
```

When you are confident in your final solution, mark it with:
FINAL_SOLUTION:
```sql
YOUR FINAL SQL QUERY HERE
```

Please start by analyzing the issue and proposing a corrected SQL query."""
    return prompt


def extract_sql_from_response(response: str) -> Optional[str]:
    """Extract SQL query from LLM response."""
    # Look for SQL code blocks
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    
    # Fallback: look for any code block
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            sql = response[start:end].strip()
            # Remove language identifier if present
            if sql.startswith("sql\n"):
                sql = sql[4:]
            return sql
    
    return None


def is_final_solution(response: str) -> bool:
    """Check if the response indicates a final solution."""
    return "FINAL_SOLUTION:" in response or "FINAL SOLUTION:" in response


def process_sample(sample: Dict, conn, model: str) -> Dict:
    """Process a single sample with iterative debugging."""
    print(f"\n{'='*60}")
    print(f"Processing: {sample['db_id']} - {sample.get('question_id', 'N/A')}")
    print(f"{'='*60}")
    
    messages = [
        {"role": "system", "content": "You are an expert SQL debugger."},
        {"role": "user", "content": build_initial_prompt(sample)}
    ]
    
    result = {
        "db_id": sample["db_id"],
        "question_id": sample.get("question_id"),
        "iterations": [],
        "final_sql": None,
        "success": False
    }
    
    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration + 1}/{MAX_ITERATIONS}")
        print(f"-" * 60)
        
        # Get LLM response
        response = call_llm(messages, model)
        if not response:
            print("❌ Failed to get LLM response")
            break
        
        print(f"LLM Response preview: {response[:150]}...")
        
        # Extract SQL from response
        sql = extract_sql_from_response(response)
        if not sql:
            print("⚠ No SQL found in response")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please provide a SQL query in a code block."})
            continue
        
        print(f"Extracted SQL: {sql[:150]}...")
        
        # Check if marked as final
        marked_as_final = is_final_solution(response)
        if marked_as_final:
            print("📍 LLM marked this as FINAL_SOLUTION")
        
        # Execute SQL
        print(f"Executing SQL...")
        exec_result = execute_sql(conn, sql)
        
        if exec_result["success"]:
            print(f"✓ Query executed successfully")
            if exec_result.get("results") is not None:
                print(f"  Returned {len(exec_result['results'])} rows")
        else:
            print(f"❌ Query failed: {exec_result['error'][:100]}")
        
        iteration_data = {
            "iteration": iteration + 1,
            "sql": sql,
            "execution_result": exec_result,
            "is_final": is_final_solution(response)
        }
        result["iterations"].append(iteration_data)
        
        # Check if this is a successful final solution
        if is_final_solution(response) and exec_result["success"]:
            result["final_sql"] = sql
            result["success"] = True
            print(f"✓ Final solution reached and executed successfully!")
            break
        
        # Prepare feedback for next iteration
        if exec_result["success"]:
            feedback = f"Query executed successfully!\n"
            if exec_result.get("results"):
                feedback += f"Results: {exec_result['results'][:3]}...\n"
                feedback += f"Columns: {exec_result['columns']}\n"
            feedback += f"Rows affected: {exec_result.get('rowcount', 0)}\n\n"
            if is_final_solution(response):
                feedback += "Great! Your FINAL_SOLUTION executed successfully."
            else:
                feedback += "If this looks correct, please provide your FINAL_SOLUTION."
        else:
            feedback = f"Query failed with error:\n{exec_result['error']}\n\n"
            if is_final_solution(response):
                feedback += "Your FINAL_SOLUTION had an error. Please fix it and try again."
            else:
                feedback += "Please fix the issue and provide an updated query."
        
        print(f"Feedback: {feedback[:150]}...")
        
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": feedback})
    
    # If no successful final solution was explicitly marked, use the last successful query
    if not result["final_sql"] and result["iterations"]:
        for iter_data in reversed(result["iterations"]):
            if iter_data["execution_result"]["success"]:
                result["final_sql"] = iter_data["sql"]
                result["success"] = True
                print(f"ℹ Using last successful query as final solution (not explicitly marked)")
                break
        
        # If still no successful query, use the last attempted query
        if not result["final_sql"]:
            result["final_sql"] = result["iterations"][-1]["sql"]
            result["success"] = False
            print(f"⚠ No successful query found - using last attempt")
    
    return result


def save_results(results: List[Dict], output_path: str):
    """Save results to JSONL file."""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BIRD-CRITIC Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name for Ollama")
    parser.add_argument("--db-host", type=str, default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port")
    parser.add_argument("--db-user", type=str, default="root", help="Database user")
    parser.add_argument("--db-password", type=str, default="123123", help="Database password")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset, args.samples)
    
    # Process each sample
    results = []
    for sample in dataset:
        # Connect to the specific database
        conn = connect_to_db(
            sample["db_id"],
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password
        )
        
        if not conn:
            print(f"Skipping {sample['db_id']} - connection failed")
            continue
        
        try:
            # Run preprocessing SQL if provided
            if "preprocess_sql" in sample and sample["preprocess_sql"]:
                print(f"Running preprocessing SQL...")
                for sql in sample["preprocess_sql"]:
                    execute_sql(conn, sql)
            
            # Process the sample
            result = process_sample(sample, conn, args.model)
            results.append(result)
            
            # Run cleanup SQL if provided
            if "clean_up_sql" in sample and sample["clean_up_sql"]:
                print(f"Running cleanup SQL...")
                for sql in sample["clean_up_sql"]:
                    execute_sql(conn, sql)
        
        finally:
            conn.close()
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"Summary: {successful}/{len(results)} samples processed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()