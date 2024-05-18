import os
# Configuration files directory
config_dir = "/workspace/raglab-exp/config/selfrag_reproduction"
# Get the algorithm name from the config directory path
algorithm_name = config_dir.split("/")[-1]

# Run scripts directory
run_dir = f"/workspace/raglab-exp/run/rag_inference/{algorithm_name}"

# Create the run scripts directory (if it doesn't exist)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# Iterate over the configuration files directory
for filename in os.listdir(config_dir):
    if filename.endswith(".yaml"):
        # Get the filename (without extension)
        basename = os.path.splitext(filename)[0]

        # Generate the Shell script filename
        script_filename = os.path.join(run_dir, f"{basename}.sh")

        # Determine the Python file to execute
        if "interact" in filename:
            python_file = "main-interact.py"
        else:
            python_file = "main-evaluation.py"

        # Generate the Shell script content
        script_content = f"# export CUDA_VISIBLE_DEVICES=0\n"
        script_content += f"python ./{python_file}\\\n"
        script_content += f"    --config ./config/{algorithm_name}/{filename}"

        # Write the Shell script file
        with open(script_filename, "w") as script_file:
            script_file.write(script_content)
        print(f"Generated {script_filename}")