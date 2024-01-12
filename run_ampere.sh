#!/bin/bash

# Define tasks
tasks=("hellaswag_nl" "arc_nl" "truthfulqa_nl" "mmlu_nl")
models_file="model_names.txt"
load_in_8bit=false
load_in_4bit=false
batch_size="auto"
declare -A failed_tasks  # Associative array to track failed tasks

# Function to show script usage
usage() {
    echo "Usage: $0 [-m <model_names_file>] [--load_in_8bit] [--load_in_4bit]"
    echo "  -m: Specify a custom file that contains model names. Default is 'model_names.txt'."
    echo "  --load_in_8bit: Include 'load_in_8bit=True' in model arguments."
    echo "  --load_in_4bit: Include 'load_in_4bit=True' in model arguments."
    echo "  -b: Batch size to use. Will use batch size finder by default."
    echo "  -h: Display this help message."
}

# Parse command-line options
while getopts "hm:b:-:" opt; do
  case $opt in
    m) models_file="$OPTARG"
       ;;
    b) batch_size="$OPTARG"
       ;;
    -) case "${OPTARG}" in
         load_in_8bit)
             load_in_8bit=true
             ;;
         load_in_4bit)
             load_in_4bit=true
             ;;
         *)
             echo "Invalid option: --${OPTARG}. Use -h for help." >&2
             exit 1
             ;;
       esac
       ;;
    h) usage
       exit 0
       ;;
    \?) echo "Invalid option: -$OPTARG. Use -h for help." >&2
        exit 1
       ;;
  esac
done

# Check if the models file exists
if [ ! -f "$models_file" ]; then
    echo "Error: Model names file '$models_file' not found."
    exit 1
fi


source .venv/bin/activate

for task in "${tasks[@]}"; do
  while IFS= read -r model_name; do
    # Skip empty lines
    if [ -z "$model_name" ]; then
        continue
    fi

    IFS='/' read -ra split <<< "$model_name"
    model_alias="${split[-1]}"
    echo "${task}: ${model_alias}"

    # Construct the model_args based on the CLI arguments
    model_args="pretrained=${model_name},use_accelerate=True,device_map_option=auto,dtype=auto"
    if [ "$load_in_8bit" = true ]; then
        model_args+=",load_in_8bit=True"
        echo "Loading ${model_name} in 8-bit"
    elif [ "$load_in_4bit" = true ]; then
        model_args+=",load_in_4bit=True"
        echo "Loading ${model_name} in 4-bit"
    fi

    echo "Processing with batch size: ${batch_size}."

    python main.py \
      --model hf-auto \
      --model_alias "$model_alias" \
      --tasks "$task" \
      --task_alias "$task" \
      --model_args "$model_args" \
      --batch_size "$batch_size"
    exit_status=$?

    # Check if the python script failed
    if [ $exit_status -ne 0 ]; then
        # Record the failed task
        failed_tasks["$task,$model_alias"]=1
    fi
  done < "$models_file"
done

# Check if any tasks failed
if [ ${#failed_tasks[@]} -eq 0 ]; then
    echo "All tests run successfully."
else
    echo "Failed task and model_name combinations:"
    for key in "${!failed_tasks[@]}"; do
        echo "$key"
    done
fi

# Copy to evaluation leaderboard and git commit to the interface
cp logs/*mmlu*.json llm_leaderboard/open_dutch_llm_leaderboard/evals/mmlu/
cp logs/*truthfulqa*.json llm_leaderboard/open_dutch_llm_leaderboard/evals/truthfulqa/
cp logs/*arc*.json llm_leaderboard/open_dutch_llm_leaderboard/evals/arc/
cp logs/*hellaswag*.json llm_leaderboard/open_dutch_llm_leaderboard/evals/hellaswag/

cd llm_leaderboard/open_dutch_llm_leaderboard/

python generate_overview_json.py

commit_str=""
while IFS= read -r line
do
    commit_str+="- [$line](https://huggingface.co/$line)\n"
done < "$models_file"

git add .
git commit -m "add evals" -m "$commit_str"
