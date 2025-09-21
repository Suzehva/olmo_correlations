import numpy as np
from analyzer_class import (
    AnalyzerClass, 
    combine_images_original_sizes,
    NEXT_TOKEN_NAME,
)

# Constants
SYSTEM_YEARS = [2004, 2054]
BASIC_PROMPT_YEARS = [1924, 2124]
BASE_PROMPT = "In [year] there"

# System prompts that exist in the cleaned up directories
SYSTEM_PROMPTS = [
    "The current date is {year}.", 
    "Current date: {year}", 
    "Today Date: {year}", 
    "The year is {year}.", 
    "It is {year}."
]

# Model pairs: (base_model, instruct_model)
MODEL_PAIRS = [
    ("meta-llama_Llama-3.1-8B", "meta-llama_Llama-3.1-8B-Instruct"),
    ("allenai_OLMo-2-1124-7B", "allenai_OLMo-2-1124-7B-Instruct"), 
    ("allenai_OLMo-2-0425-1B", "allenai_OLMo-2-0425-1B-Instruct")
]

def generate_prompt_name(system_prompt, system_year):
    """Generate the prompt name (without model_predictions__ prefix)."""
    formatted_prompt = system_prompt.format(year=system_year)
    full_prompt = f"{formatted_prompt} {BASE_PROMPT}"
    
    # Clean for directory name
    prompt_name = full_prompt.replace(' ', '_').replace('[', '_').replace(']', '_').replace('/', '_')
    return prompt_name

def print_ce_analysis(ce_results):
    """Print organized CE analysis."""
    print("\n" + "="*80)
    print("CROSS-ENTROPY ANALYSIS SUMMARY")
    print("="*80)
    
    # 1. Average CE per system prompt per year
    print("\n1. AVERAGE CE LOSS PER SYSTEM PROMPT PER YEAR (across all models):")
    print("-" * 60)
    
    prompt_year_stats = {}
    for key, ce_value in ce_results.items():
        model_type, prompt_desc, system_year = key
        prompt_key = (prompt_desc, system_year)
        if prompt_key not in prompt_year_stats:
            prompt_year_stats[prompt_key] = []
        prompt_year_stats[prompt_key].append(ce_value)
    
    for (prompt_desc, system_year), ce_values in sorted(prompt_year_stats.items()):
        avg_ce = sum(ce_values) / len(ce_values)
        print(f"  {prompt_desc} (year={system_year}): {avg_ce:.4f} (±{np.std(ce_values):.4f})")
    
    # 2. Average CE per model type
    print("\n2. AVERAGE CE LOSS PER MODEL TYPE (across all prompts and years):")
    print("-" * 60)
    
    model_type_stats = {}
    for key, ce_value in ce_results.items():
        model_type, prompt_desc, system_year = key
        if model_type not in model_type_stats:
            model_type_stats[model_type] = []
        model_type_stats[model_type].append(ce_value)
    
    for model_type, ce_values in sorted(model_type_stats.items()):
        avg_ce = sum(ce_values) / len(ce_values)
        print(f"  {model_type}: {avg_ce:.4f} (±{np.std(ce_values):.4f}) [n={len(ce_values)}]")
    
    # 3. Instruct vs Base comparison
    print("\n3. AVERAGE CE LOSS: INSTRUCT vs BASE MODELS:")
    print("-" * 60)
    
    instruct_ces = []
    base_ces = []
    for key, ce_value in ce_results.items():
        model_type, prompt_desc, system_year = key
        if "instruct" in model_type.lower():
            instruct_ces.append(ce_value)
        else:
            base_ces.append(ce_value)
    
    if instruct_ces:
        avg_instruct = sum(instruct_ces) / len(instruct_ces)
        print(f"  Instruct models: {avg_instruct:.4f} (±{np.std(instruct_ces):.4f}) [n={len(instruct_ces)}]")
    
    if base_ces:
        avg_base = sum(base_ces) / len(base_ces)
        print(f"  Base models: {avg_base:.4f} (±{np.std(base_ces):.4f}) [n={len(base_ces)}]")
    
    # 4. Best and worst performing combinations
    print("\n4. BEST AND WORST PERFORMING COMBINATIONS:")
    print("-" * 60)
    
    sorted_results = sorted(ce_results.items(), key=lambda x: x[1])
    print("  Best (lowest CE):")
    for i in range(min(5, len(sorted_results))):
        key, ce_value = sorted_results[i]
        model_type, prompt_desc, system_year = key
        print(f"    {model_type} | {prompt_desc} | year={system_year}: {ce_value:.4f}")
    
    print("  Worst (highest CE):")
    for i in range(max(0, len(sorted_results)-5), len(sorted_results)):
        key, ce_value = sorted_results[i]
        model_type, prompt_desc, system_year = key
        print(f"    {model_type} | {prompt_desc} | year={system_year}: {ce_value:.4f}")

def analyze_system_prompts(analyzer):
    """Analyze system prompts with the new simplified structure."""
    ce_results = {}  # (model_type, prompt_desc, system_year) -> ce_value
    
    for base_model, instruct_model in MODEL_PAIRS:
        model_name = base_model.split("/")[-1].lower()
        
        for system_year in SYSTEM_YEARS:
            for system_prompt in SYSTEM_PROMPTS:
                # Generate prompt name and display name
                prompt_name = generate_prompt_name(system_prompt, system_year)
                prompt_display = f"{system_prompt.format(year=system_year)} {BASE_PROMPT}"
                
                # Test both base and instruct models
                for model_clean, model_type, checkpoint in [
                    (base_model.replace("/", "_"), f"{model_name}_base", "final"),
                    (instruct_model.replace("/", "_"), f"{model_name}_instruct", "final_instruct")
                ]:
                    # Load predictions
                    prompt_preds = analyzer.load_other_prompts([prompt_name], [model_clean], all_verbs=False, wordboundary=False, stored_on_disk=True)
                    
                    # Create bar plot (use the full model identifier for proper display names)
                    data_type_with_prompt = f"{NEXT_TOKEN_NAME}\n{prompt_display}"
                    analyzer.bar_plot(prompt_preds[prompt_name][model_clean], model_clean, data_type_with_prompt, checkpoint,
                                    BASIC_PROMPT_YEARS[0], BASIC_PROMPT_YEARS[1], make_relative=False, system_year=system_year, folder_name_proposal="instruct_vs_base")
                    
                    # Compute cross-entropy
                    ce = analyzer.compute_cross_entropy_over_range(prompt_preds[prompt_name][model_clean], model_clean, checkpoint,
                                                                 BASIC_PROMPT_YEARS[0], BASIC_PROMPT_YEARS[1], gold_dist_year=system_year)
                    
                    # Store result
                    ce_results[(model_type, prompt_display, system_year)] = ce['average_loss']
    
    # Generate combined plots
    for base_model, _ in MODEL_PAIRS:
        model_name = base_model.split("/")[-1].lower()
        
        layout = []
        for system_prompt in SYSTEM_PROMPTS:
            row = []
            for system_year in SYSTEM_YEARS:
                prompt_display = f"{system_prompt.format(year=system_year)} {BASE_PROMPT}"
                
                # Generate filenames to match the actual saved files (keep [year] pattern)
                base_model_clean = base_model.replace("/", "_")
                instruct_model_clean = instruct_model.replace("/", "_")
                instruct_filename = f"{instruct_model_clean}_checkpointfinal_instruct_Next-token_predictions_{prompt_display}_{BASIC_PROMPT_YEARS[0]}-{BASIC_PROMPT_YEARS[1]}.png"
                base_filename = f"{base_model_clean}_checkpointfinal_Next-token_predictions_{prompt_display}_{BASIC_PROMPT_YEARS[0]}-{BASIC_PROMPT_YEARS[1]}.png"
                
                # Clean only spaces and slashes, keep [year] brackets
                instruct_filename = instruct_filename.replace(' ', '_').replace('/', '_')
                base_filename = base_filename.replace(' ', '_').replace('/', '_')
                
                row.extend([instruct_filename, base_filename])
            layout.append(row)
        
        combine_images_original_sizes(layout, "instruct_vs_base", f"instruct_vs_base/{base_model.replace('/', '_')}.png")
    
    # Print analysis
    print_ce_analysis(ce_results)

if __name__ == "__main__":
    analyzer = AnalyzerClass()
    analyze_system_prompts(analyzer) 