import asyncio
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional
import random
from datetime import datetime
import uuid
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm.asyncio import tqdm
import litellm
from dotenv import load_dotenv
import polars as pl
import aiofiles
from litellm.utils import ModelResponse

class IOIEvaluator:
    def __init__(self, model_id: str, api_base: Optional[str] = None,  subset: Optional[str] = None,
                 num_generations: int = 50, num_retries: int = 10, 
                 concurrency: int = 10, num_problems: Optional[int] = None, 
                 last_subtask: bool = False, dry_run: bool = False,
                 override: bool = False, model_postfix: Optional[str] = None,
                 revision: Optional[str] = None, timeout: Optional[int] = 600,
                 use_requests: bool = False, max_tokens: Optional[int] = None, hub_dataset_id: Optional[str] = None):
        self.hub_dataset_id = hub_dataset_id
        self.model_id = model_id
        self.api_base = api_base
        self.subset = subset
        self.num_generations = num_generations
        self.num_retries = num_retries
        self.concurrency = concurrency
        self.num_problems = num_problems
        self.last_subtask = last_subtask
        self.dry_run = dry_run
        self.override = override
        self.revision = revision
        # Create organization and model directories
        self.timeout = timeout
        self.use_litellm = not use_requests
        self.max_tokens = max_tokens
        
        # Tracking totals
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.model_postfix = model_postfix
        
        # Semaphore for controlling concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        
        # HTTP session for direct API calls when not using litellm
        self._session = None

        if self.api_base:
            logger.info(f"Using API base: {self.api_base}")
            
        if not self.use_litellm:
            logger.info("Using direct asyncio requests instead of LiteLLM")

        if dry_run:
            logger.warning("Running in dry-run mode - no actual LLM calls will be made")

        # Create results directory
        self.model_dir = Path("results") / self.get_model_name() if self.hub_dataset_id is None else Path("results") / self.hub_dataset_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # File path for the single JSONL file
        self.results_file = self.model_dir / "results.jsonl"
        
        # Lock for file access
        self._file_lock = asyncio.Lock()

    async def save_result_locally(self, result: Dict, year: int, problem_id: str, subtask: str, solution_number: int):
        """Save a single result to local JSONL storage with locking."""
        # Ensure problem_id is included in the result
        result['year'] = year
        result['problem_id'] = problem_id
        result['subtask'] = subtask
        result['solution_number'] = solution_number
        
        try:
            # Use lock to prevent concurrent writes
            async with self._file_lock:
                async with aiofiles.open(self.results_file, 'a') as f:
                    await f.write(json.dumps(result) + '\n')
        except Exception as e:
            logger.error(f"Failed to save result locally: {str(e)}")

    async def load_previous_results(self) -> Optional[pl.DataFrame]:
        """Load previous results from both HuggingFace Hub and local JSONL storage."""
        if self.override:
            logger.info("Override mode enabled - not loading previous results")
            return None
        
        results_dfs = []
        
        # Try loading from Hub
        repo_name = self.hub_dataset_id if self.hub_dataset_id is not None else self.get_model_name()
        try:
            logger.info(f"Attempting to load previous results from HuggingFace Hub: {repo_name}")
            dataset = load_dataset(repo_name, split="train")
            if dataset is not None:
                # Convert to pandas then to polars
                df = dataset.to_polars()

                # Add a column indicating if the result is local
                df = df.with_columns([
                    pl.lit(False).alias('is_local')
                ])
                results_dfs.append(df)

                logger.info(f"Loaded {len(df)} previous results from HuggingFace Hub")
        except Exception as e:
            logger.info(f"Could not load from HuggingFace Hub: {str(e)}")

        # Try loading from local storage
        try:
            if self.results_file.exists():
                results = []
                async with self._file_lock:
                    async with aiofiles.open(self.results_file, 'r') as f:
                        async for line in f:
                            try:
                                result = json.loads(line.strip())
                                results.append(result)
                            except Exception as e:
                                logger.error(f"Failed to parse JSONL line: {str(e)}")
            
                if results:
                    local_df = pl.DataFrame(results).with_columns([
                        pl.lit(True).alias('is_local')
                    ])
                    results_dfs.append(local_df)
                    logger.info(f"Loaded {len(local_df)} previous results from local storage")
        except Exception as e:
            logger.error(f"Failed to load from local storage: {str(e)}")

        # Combine results if we have any
        if results_dfs:
            # Select just columns: 'generation', 'code', 'language', 'model_kwargs', 'metadata', 'uuid', 'problem_id', 'subtask', 'solution_number', 'is_local'
            common_columns = ['generation', 'code', 'language', 'model_kwargs', 'metadata', 'uuid', 'year', 'problem_id', 'subtask', 'solution_number', 'is_local']

            # Add missing 'year' column with None values if needed
            results_dfs = [df if 'year' in df.columns else df.with_columns(pl.lit(None).alias('year')) for df in results_dfs]
            
            # Drop that are not in common_columns
            results_dfs = [df.select(common_columns) for df in results_dfs]

            # Try this instead:
            # Add stop_reason to metadata if it doesn't exist
            results_dfs = [df.with_columns(pl.when(pl.col('metadata').is_not_null()).then(pl.col('metadata').map_elements(lambda x: {"stop_reason": "unknown"} | x)).otherwise(pl.col('metadata')).alias('metadata')) for df in results_dfs]
            
            # Concatenate the aligned dataframes
            combined_df = pl.concat(results_dfs, how="vertical")
            
            # First sort by whether code exists (True first), then by source (local first)
            # This ensures we keep entries with code when deduplicating
            deduplicated_df = (
                combined_df
                .with_columns([
                    # Add a column indicating if code exists and is non-empty
                    pl.when((pl.col('code').is_not_null()) & (pl.col('code') != ""))
                    .then(1)
                    .otherwise(0)
                    .alias('has_code'),
                ])
                # Sort by has_code (descending) and is_local (descending)
                .sort(['has_code', 'is_local'], descending=[True, True])
                # Keep first occurrence after sorting (prioritizing entries with code and local source)
                .unique(
                    subset=["year", "problem_id", "subtask", "solution_number"],
                    keep='first'
                )
                # Drop the temporary columns
                .drop(['has_code', 'is_local'])
            )
            
            logger.info(f"Combined and deduplicated results: {len(deduplicated_df)} entries")
            return deduplicated_df
        
        return None

    def get_dummy_response(self, prompt: str, seed: int) -> Dict:
        """Generate a dummy response for dry runs."""
        dummy_code = """```cpp
int main() {
    // This is a dummy solution
    return 0;
}
```"""
        return {
            "generation": f"This is a dummy response for testing purposes.\n{dummy_code}",
            "code": "int main() {\n    // This is a dummy solution\n    return 0;\n}",
            "language": "cpp",
            "model_kwargs": {
                "seed": seed,
            },
            "metadata": {
                "usage": {
                    'completion_tokens': 10,
                    'prompt_tokens': len(prompt.split()),
                    'total_tokens': len(prompt.split()) + 10,
                    'cost': 0.0
                },
                "timestamp": datetime.now().isoformat(),
                "stop_reason": "length"  # Add stop reason for dummy response
            }
        }

    def extract_code(self, text: str) -> tuple[str, str]:
        """Extract code from the response between ```cpp and ``` markers."""
        try:
            parts = text.split("```cpp\n")
            if len(parts) > 1:
                code_block = parts[-1].split("```")[0]
                code = code_block.strip()
                if not code:
                    logger.warning("Empty code block found")
                    return "", "cpp"
                return code, "cpp"
            logger.warning("No code block found in the response")
            return "", "unknown"
        except Exception as e:
            logger.error(f"Failed to extract code: {str(e)}")
            return "", "unknown"

    async def generate_completion(self, prompt: str, seed: int) -> Dict:
        """Generate completion using direct asyncio HTTP requests."""
        retry_budget = self.num_retries
        
        while retry_budget > 0:
            try:
                await asyncio.sleep(random.uniform(0.0, 0.1))
                async with self._session.post(
                    f"{self.api_base}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [{"role": "user", "content": prompt}],
                        "seed": seed,
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "max_tokens": self.max_tokens,
                    },
                    headers={"Authorization": "Bearer EMPTY"},
                ) as response:
                    result = await response.json(content_type=None)
                    
                    if result is None:
                        logger.error("Received None response from API")
                        retry_budget -= 1
                        await asyncio.sleep(5)
                        continue
                    
                    # Extract response content
                    message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Extract token usage
                    usage = result.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    # Update totals
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    
                    # Extract code
                    code, language = self.extract_code(message_content)
                    
                    response_dict = {
                        "generation": message_content,
                        "code": code,
                        "language": language,
                        "model_kwargs": {
                            "seed": seed,
                        },
                        "metadata": {
                            "usage": {
                                'completion_tokens': completion_tokens,
                                'prompt_tokens': prompt_tokens,
                                'total_tokens': total_tokens,
                            },
                            "timestamp": datetime.now().isoformat(),
                            "stop_reason": result.get("choices", [{}])[0].get("finish_reason", "unknown")
                        }
                    }
                    
                    
                    return response_dict
                    
            except Exception as e:
                logger.exception(f"API error (will retry): {e}")
                retry_budget -= 1
                await asyncio.sleep(10)

        raise Exception("All retries failed for direct API call")
                

    async def call_llm(self, prompt: str, seed: int) -> Dict:
        """Call the LLM using LiteLLM's built-in retry mechanism or direct asyncio requests."""
        if self.dry_run:
            result = self.get_dummy_response(prompt, seed)
            return result
            
        if not self.use_litellm:
            return await self.generate_completion(prompt, seed)

        return await self.call_litellm(prompt, seed)

    async def call_litellm(self, prompt: str, seed: int) -> Dict:
        model_name = self.model_id
        kwargs = {}
        if self.model_id.startswith("sglang/"):
            model_name = model_name.replace("sglang/", "custom_openai/")
            kwargs["api_base"] = self.api_base
            kwargs["api_key"] = "sk-proj-1234567890"

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response: ModelResponse = await litellm.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": prompt, "cache_control": {"type": "ephemeral"}}],
            seed=seed,
            num_retries=self.num_retries,
            top_p=0.8,
            temperature=0.7,
            timeout=self.timeout,
            **kwargs
        )

        # Extract stop reason
        stop_reason = response.choices[0].finish_reason
        
        # Extract usage information safely
        usage = {}
        cost = 0.0
        if hasattr(response, 'usage'):
            try:
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                total_tokens = getattr(response.usage, 'total_tokens', 0)
                
                # Calculate cost using litellm
                try:
                    cost = litellm.completion_cost(completion_response=response)
                except Exception as e:
                    logger.warning(f"Failed to calculate cost: {str(e)}")
                    cost = 0.0
                
                usage = {
                    'completion_tokens': completion_tokens,
                    'prompt_tokens': prompt_tokens,
                    'total_tokens': total_tokens,
                    'cost': cost
                }
                
                # Update totals
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_cost += cost
                
            except Exception as e:
                logger.error(f"Failed to extract usage information: {str(e)}")
        
        message_content = response.choices[0].message.content if response.choices else ""
        
        # Extract code from the response
        code, language = self.extract_code(message_content or "")
        
        result = {
            "generation": message_content,
            "code": code,
            "language": language,
            "model_kwargs": {
                "seed": seed,
            },
            "metadata": {
                "usage": usage,
                "timestamp": datetime.now().isoformat(),
                "stop_reason": stop_reason
            }
        }
        return result

    async def create_solution_requests(self, subtasks: List[Dict]) -> List[Dict]:
        """Prepare result entries for a single problem."""
        results = []
        for subtask in subtasks:
            prompt = subtask['problem']
            for i in range(self.num_generations):
                try:
                    random_uuid = str(uuid.uuid4())
                    
                    results.append({
                        "year": subtask['year'],
                        "problem_id": subtask['id'],
                        "subtask": subtask["subtask"],
                        "prompt": prompt,
                        "generation": None,
                        "code": "",
                        "language": "unknown",
                        "solution_number": i,
                        "uuid": random_uuid,
                        "model_kwargs": {"seed": i},
                        "metadata": {
                            "usage": {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0, 'cost': 0.0},
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to prepare prompts for problem {subtask['id']}, subtask {subtask['subtask']}: {str(e)}")
                    return []

        return results

    async def run_evaluation(self):
        """Run the evaluation for all problems."""
        try:
            # Create HTTP session if using direct API calls
            if not self.use_litellm and not self.dry_run:
                import aiohttp
                self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout), connector=aiohttp.TCPConnector(limit=self.concurrency, ttl_dns_cache=300, keepalive_timeout=self.timeout))
                
            
            logger.info(f"Loading IOI dataset for subset: {self.subset}")
            dataset = load_dataset("open-r1/ioi", split=self.subset)
            problem_subtasks = defaultdict(list)
            for problem in dataset:
                problem_subtasks[(problem["year"], problem["id"])].append(problem)
            problem_ids = list(problem_subtasks.keys())
            if self.num_problems is not None:
                problem_ids = problem_ids[:self.num_problems]
                logger.info(f"Limited evaluation to first {self.num_problems} problems")
            
            logger.info(f"Starting evaluation of {len(problem_ids)} problems...")

            # Step 1: Generate all solution requests
            all_solution_requests = []
            for problem_id in tqdm(problem_ids, desc="Preparing solution requests"):
                subtasks = problem_subtasks[problem_id]
                if self.last_subtask:
                    subtasks = [subtasks[-1]]
                requests = await self.create_solution_requests(subtasks)
                all_solution_requests.extend(requests)
            
            # Convert to Polars DataFrame for efficient operations
            requests_df = pl.DataFrame(all_solution_requests)
            logger.info(f"Created {len(requests_df)} solution requests")

            # Step 2: Load previous results
            previous_df = None
            if not self.override:
                previous_df = await self.load_previous_results()
                if previous_df is not None:
                    logger.info(f"Loaded {len(previous_df)} previous results")
            
            
            # Step 3: Merge solution requests with previous results efficiently
            if previous_df is not None:
                # Keep only the columns we want to preserve from previous results
                preserve_cols = ['generation', 'code', 'language', 'metadata', 'model_kwargs']

                preserve_cols_with_key = preserve_cols + ['year', 'problem_id', 'subtask', 'solution_number']
                previous_df = previous_df.select(preserve_cols_with_key).filter(pl.col('generation').is_not_null() & (pl.col('generation') != ""))
                
                # Merge using polars, keeping all solution requests and only matching previous results
                merged_df = requests_df.join(
                    previous_df,
                    on=('year', 'problem_id', 'subtask', 'solution_number'),
                    how='left',
                    suffix='_prev'
                )

                # Update values from previous results where they exist
                for col in preserve_cols:
                    prev_col = f'{col}_prev'
                    merged_df = merged_df.with_columns(
                        pl.when(pl.col(prev_col).is_not_null())
                        .then(pl.col(prev_col))
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
                
                # Drop the _prev columns
                merged_df = merged_df.select([
                    c for c in merged_df.columns if not c.endswith('_prev')
                ])
            else:
                merged_df = requests_df

            # Count how many need to be generated
            to_generate_df = merged_df.filter(
                (pl.col('generation').is_null()) | 
                (pl.col('generation') == "")
            )

            # Update seeds ensuring uniqueness
            to_generate_dicts = to_generate_df.to_dicts()
            logger.info(f"Need to generate {len(to_generate_df)} out of {len(merged_df)} total entries")

            if len(to_generate_df) == 0:
                logger.info("No generations needed - all results are already available")
                return

            # Run generations for entries without results
            async def process_single(row: Dict) -> Dict:
                async with self._semaphore:
                    try:
                        llm_result = await self.call_llm(
                            row["prompt"], 
                            row["model_kwargs"]["seed"]
                        )
                        
                        # Log progress and token usage
                        if llm_result["metadata"].get("usage"):
                            usage = llm_result["metadata"]["usage"]
                            logger.info(
                                f"Problem {row['problem_id']} (Solution {row['solution_number']}) - "
                                f"Tokens: {usage.get('total_tokens', 0)} "
                                f"(prompt: {usage.get('prompt_tokens', 0)}, "
                                f"completion: {usage.get('completion_tokens', 0)}) - "
                                f"Cost: ${usage.get('cost', 0.0):.4f}"
                            )
                        
                        llm_result["uuid"] = row["uuid"]

                        # Save result immediately
                        await self.save_result_locally(llm_result, row["year"], row["problem_id"], row["subtask"], row["solution_number"])

                        return llm_result
                    except Exception as e:
                        logger.error(f"Failed generation for problem {row['problem_id']}: {str(e)}")
                        error_result = {
                            "generation": "",
                            "code": "",
                            "language": "unknown",
                            "uuid": row["uuid"],
                            "metadata": {
                                "error": str(e),
                                "usage": {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0, 'cost': 0.0},
                                "timestamp": datetime.now().isoformat(),
                                "stop_reason": "error"  # Add stop reason for error case
                            }
                        }
                        return error_result

            # Run generations in parallel with controlled concurrency
            tasks = [process_single(row) for row in to_generate_dicts]
            generated_results = await tqdm.gather(*tasks, desc="Running generations")

            # Convert generated results to DataFrame and update original DataFrame
            generated_df = pl.DataFrame(generated_results)

            # Merge generated results with previous results
            merged_df = merged_df.join(
                generated_df,
                on='uuid',
                how='left',
                suffix='_gen'
            )

            # Update the old columns with the new values
            for col in ['generation', 'code', 'language', 'metadata', 'model_kwargs']:
                merged_df = merged_df.with_columns(
                    pl.when(pl.col(f'generation_gen').is_not_null() & (pl.col(f'generation_gen') != ""))
                    .then(pl.col(f'{col}_gen'))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

            # Drop the _gen columns
            merged_df = merged_df.select([
                c for c in merged_df.columns if not c.endswith('_gen')
            ])
                
            # Validate results before pushing to hub
            valid_results = merged_df.filter(
                (pl.col('generation').is_not_null()) & 
                (pl.col('generation') != "")
            )

            total_expected = len(merged_df)
            total_valid = len(valid_results)

            logger.info(f"Valid results: {total_valid}/{total_expected}")

            # Only push to hub if all results are valid
            if total_valid == total_expected:
                # Convert to HF Dataset
                output_dataset = Dataset.from_polars(merged_df)
                model_name = self.get_model_name()
                
                try:
                    hub_dataset_id = self.hub_dataset_id if self.hub_dataset_id is not None else model_name
                    output_dataset.push_to_hub(hub_dataset_id)
                    logger.info(f"Pushed to hub: {hub_dataset_id}")
                except Exception as e:
                    logger.error(f"Failed to push to hub: {str(e)}")
            else:
                logger.warning(
                    f"Not pushing to hub - missing {total_expected - total_valid} valid results. "
                    "Results saved locally and can be retried later."
                )

            # Log final statistics
            # logger.info(f"Evaluation completed. Total successful generations: {successful}/{len(all_results)}")
            logger.info(
                f"Total tokens used: {self.total_prompt_tokens + self.total_completion_tokens} "
                f"(prompt: {self.total_prompt_tokens}, completion: {self.total_completion_tokens})"
            )
            logger.info(f"Total cost: ${self.total_cost:.4f}")
            
            # Clean up HTTP session if using direct API calls
            if self._session is not None:
                await self._session.close()
                self._session = None
                
            return merged_df
        except Exception as e:
            # Clean up HTTP session if using direct API calls
            if self._session is not None:
                await self._session.close()
                self._session = None
            raise e

    
    def get_model_name(self):
        model_name = f"ioi-eval-{self.model_id.replace('/', '_')}"
        if self.dry_run:
            model_name = f"dummy-{model_name}"

        if self.revision:
            model_name = f"{model_name}-{self.revision.replace('/', '_')}"

        if self.model_postfix:
            model_name = f"{model_name}-{self.model_postfix}"

        return model_name


def main():
    load_dotenv()  # Load environment variables from .env file
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLMs on IOI problems")
    parser.add_argument("--model_id", required=True, help="Model ID")
    parser.add_argument("--api_base", help="API base URL for the model")
    parser.add_argument("--subset", default="test", help="IOI subset to generate solutions for (train or test)")
    parser.add_argument("--num_generations", type=int, default=50, help="Number of generations per problem")
    parser.add_argument("--num_retries", type=int, default=10, help="Number of retries for failed API calls")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent generations")
    parser.add_argument("--num_problems", type=int, default=None, help="Number of problems to evaluate (None for all)")
    parser.add_argument("--last_subtask", action="store_true", help="Only evaluate the last subtask for each problem (usually the full problem)")
    parser.add_argument("--dry_run", action="store_true", help="Run without making actual LLM calls")
    parser.add_argument("--override", action="store_true", help="Override existing results and start fresh")
    parser.add_argument("--model_postfix", help="Postfix for the model name")
    parser.add_argument("--revision", help="Revision to use for the model")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for the LLM call")
    parser.add_argument("--use_requests", action="store_true", default=False, help="Use requests instead of litellm")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--hub_dataset_id", type=str, default=None, help="Hub dataset ID to push results to. If `None`, will push results to user's namespace.")
    args = parser.parse_args()

    evaluator = IOIEvaluator(
        model_id=args.model_id,
        api_base=args.api_base,
        subset=args.subset,
        num_generations=args.num_generations,
        num_retries=args.num_retries,
        concurrency=args.concurrency,
        num_problems=args.num_problems,
        last_subtask=args.last_subtask,
        dry_run=args.dry_run,
        override=args.override,
        model_postfix=args.model_postfix,
        revision=args.revision,
        timeout=args.timeout,
        use_requests=args.use_requests,
        max_tokens=args.max_tokens,
        hub_dataset_id=args.hub_dataset_id
    )
    asyncio.run(evaluator.run_evaluation())

if __name__ == "__main__":
    main() 