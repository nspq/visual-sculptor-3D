import subprocess
import os

def run_openmvg_global_pipeline(input_dir, output_dir):
    # Run SfM_GlobalPipeline.py script
    script_path = "openMVG/SfM_GlobalPipeline.py"
    command = [
        "python", script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir
    ]

    try:
        subprocess.run(command, check=True)
        print("OpenMVG Global Pipeline completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running OpenMVG Global Pipeline: {e}")

if __name__ == "__main__":
    # Specify the path to your input directory and output directory
    input_directory = "/data"
    output_directory = "/outputs"

    run_openmvg_global_pipeline(input_directory, output_directory)
