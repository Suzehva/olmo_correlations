from analyzer_class import (
    AnalyzerClass, 
    plot_cross_entropies_per_year_over_checkpoints, plot_average_cross_entropies_across_year_prompts, plot_cross_entropies, combine_images_original_sizes,
    CO_OCCURR_NAME, EXACT_STRING_MATCH_NAME, NGRAM_NAME, NEXT_TOKEN_NAME,
    OLMO_CHECKPOINTS, PYTHIA_CHECKPOINTS,
    PROMPT_DISPLAY_NAMES,
)

def bar_plot_predictions_training_data(analyzer):
    cp = 10000
    start_year = 1950
    end_year = 2050

    # plot_training_data
    analyzer.bar_plot(analyzer.olmo_co_occurrence, "olmo", CO_OCCURR_NAME, cp, start_year, end_year)
    analyzer.bar_plot(analyzer.olmo_exact_string_match, "olmo", EXACT_STRING_MATCH_NAME, cp, start_year, end_year)
    analyzer.bar_plot(analyzer.pythia_co_occurrence, "pythia", CO_OCCURR_NAME, cp, start_year, end_year)
    analyzer.bar_plot(analyzer.pythia_exact_string_match, "pythia", EXACT_STRING_MATCH_NAME, cp, start_year, end_year)
    analyzer.bar_plot(analyzer.olmo_relative_ngram, "olmo", NGRAM_NAME, cp, start_year, end_year)
    analyzer.bar_plot(analyzer.pythia_relative_ngram, "pythia", NGRAM_NAME, cp, start_year, end_year)

    # plot_model_predictions
    analyzer.bar_plot(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, cp, start_year, end_year)
    analyzer.bar_plot(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, cp, start_year, end_year)

def plot_other_model_predictions(analyzer):
    start_year = 1950
    end_year = 2050

    # plot_model_predictions with other models
    for model_name in analyzer.other_model_predictions.keys():
        analyzer.bar_plot(analyzer.other_model_predictions[model_name], model_name, NEXT_TOKEN_NAME, "final", start_year, end_year, make_relative=False, separate_present_future=True)

def plot_other_prompt_predictions(analyzer):
    # Example (commented): Load prompt-based all-verbs predictions using good-verbs mapping
    prompts = [
        "During__year__there", "In__year__the_choir", "In__year__there", 
        "In__year__they", "In__year_,_at_the_dinner_table,_the_family", 
        "In__year_,_there", "In__year_,_with_a_knife,_he", "In__year_,_with_a_pen_to_paper,_she", 
        "In__year_,_with_his_credit_card,_he", 
        "In_the_magic_show_in__year_,_there_magically",
    ]
    models_for_prompts = ["allenai_OLMo-2-0425-1B", "EleutherAI_pythia-1.4b-deduped"]
    prompt_preds = analyzer.load_other_prompts(prompts, models_for_prompts, all_verbs=True)
    # Plot absolute (non-relative) bars for each model at the synthetic 'final' checkpoint
    for model_name in models_for_prompts:
        for prompt in prompts:
            display_prompt = PROMPT_DISPLAY_NAMES.get(prompt, prompt)
            data_type_with_prompt = f"{NEXT_TOKEN_NAME}\n{display_prompt}"
            analyzer.bar_plot(prompt_preds[prompt][model_name], model_name, data_type_with_prompt, "final", start_year, end_year, make_relative=False)
        # Saved under folder f"{model_name}_checkpointfinal"

def ce_over_years(analyzer):
    start_year = 1950
    end_year = 2050
    cp = 10000

    # compute losses for 10k checkpoint
    olmo_pred_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_predictions, "olmo", cp, start_year, end_year)
    olmo_co_occurrence_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_co_occurrence, "olmo", cp, start_year, end_year)
    olmo_ngram_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_relative_ngram, "olmo", cp, start_year, end_year)
    olmo_exact_string_match_loss = analyzer.compute_cross_entropy_over_range(analyzer.olmo_exact_string_match, "olmo", cp, start_year, end_year, allow_missing_data=True)
    print(f"Olmo predictions average loss: {olmo_pred_loss['average_loss']}")
    print(f"Olmo co-occurrence average loss: {olmo_co_occurrence_loss['average_loss']}")
    print(f"Olmo n-gram average loss: {olmo_ngram_loss['average_loss']}")
    print(f"Olmo exact string match average loss (years used: {len(olmo_exact_string_match_loss['years_used'])}): {olmo_exact_string_match_loss['average_loss']}")
    print("--------------------------------")
    
    pythia_pred_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_predictions, "pythia", cp, start_year, end_year)
    pythia_co_occurrence_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_co_occurrence, "pythia", cp, start_year, end_year)
    pythia_ngram_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_relative_ngram, "pythia", cp, start_year, end_year)
    pythia_exact_string_match_loss = analyzer.compute_cross_entropy_over_range(analyzer.pythia_exact_string_match, "pythia", cp, start_year, end_year, allow_missing_data=True)
    print(f"Pythia predictions average loss: {pythia_pred_loss['average_loss']}")
    print(f"Pythia co-occurrence average loss: {pythia_co_occurrence_loss['average_loss']}")
    print(f"Pythia n-gram average loss: {pythia_ngram_loss['average_loss']}")
    print(f"Pythia exact string match average loss (years used: {len(pythia_exact_string_match_loss['years_used'])}): {pythia_exact_string_match_loss['average_loss']}")
    print("--------------------------------")
    # print summary of average losses for 10k checkpoint
    olmo_exact_str_years = analyzer.collect_years_with_no_data(analyzer.olmo_exact_string_match, cp, start_year, end_year, "OLMo exact string match")
    olmo_pred_loss_less_data = analyzer.compute_cross_entropy_over_range(analyzer.olmo_predictions, "olmo", cp, start_year, end_year, specific_years=olmo_exact_str_years)
    print(f"Olmo predictions average loss (years used: {len(olmo_exact_str_years)}): {olmo_pred_loss_less_data['average_loss']}")
    print("--------------------------------")
    
    # print summary of loss if only using ones frome xact string match for each metods
    pythia_exact_str_years = analyzer.collect_years_with_no_data(analyzer.pythia_exact_string_match, cp, start_year, end_year, "Pythia exact string match")
    pythia_pred_loss_less_data = analyzer.compute_cross_entropy_over_range(analyzer.pythia_predictions, "pythia", cp, start_year, end_year, specific_years=pythia_exact_str_years)
    print(f"Pythia predictions average loss (years used: {len(pythia_exact_str_years)}): {pythia_pred_loss_less_data['average_loss']}")
    print("--------------------------------")

    # plot cross-entropy losses
    plot_cross_entropies([olmo_pred_loss, olmo_co_occurrence_loss, olmo_ngram_loss], [NEXT_TOKEN_NAME, CO_OCCURR_NAME, NGRAM_NAME], "olmo", start_year, end_year)
    plot_cross_entropies([pythia_pred_loss, pythia_co_occurrence_loss, pythia_ngram_loss], [NEXT_TOKEN_NAME, CO_OCCURR_NAME, NGRAM_NAME], "pythia", start_year, end_year)
    
def bar_plot_multiple_checkpoints(analyzer):
    start_year = 1950
    end_year = 2050

    analyzer.bar_plots_for_checkpoints(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, [1000, 3000, 7000], 1, 3, start_year, end_year, subplot_width=4.5, subplot_height=2.6)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, [1000, 3000, 7000], 1, 3, start_year, end_year, subplot_width=4.5, subplot_height=2.6)
    
    # Generate checkpoint grid plots (APPENDIX)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, OLMO_CHECKPOINTS, 8, 5, start_year, end_year)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_co_occurrence, "olmo", CO_OCCURR_NAME, OLMO_CHECKPOINTS, 8, 5, start_year, end_year)
    analyzer.bar_plots_for_checkpoints(analyzer.olmo_relative_ngram, "olmo", NGRAM_NAME, OLMO_CHECKPOINTS, 8, 5, start_year, end_year)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, PYTHIA_CHECKPOINTS, 2, 5, start_year, end_year)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_co_occurrence, "pythia", CO_OCCURR_NAME, PYTHIA_CHECKPOINTS, 2, 5, start_year, end_year)
    analyzer.bar_plots_for_checkpoints(analyzer.pythia_relative_ngram, "pythia", NGRAM_NAME, PYTHIA_CHECKPOINTS, 2, 5, start_year, end_year)
    
def ce_over_checkpoints(analyzer):
    start_year = 1950
    end_year = 2050

    # Plot average cross-entropies over checkpoints
    olmo_distributions = {
        NEXT_TOKEN_NAME: analyzer.olmo_predictions,
        CO_OCCURR_NAME: analyzer.olmo_co_occurrence,
        NGRAM_NAME: analyzer.olmo_relative_ngram,
    }
    plot_average_cross_entropies_over_checkpoints(analyzer, olmo_distributions, "olmo", OLMO_CHECKPOINTS, start_year, end_year)
    
    pythia_distributions = {
        NEXT_TOKEN_NAME: analyzer.pythia_predictions,
        CO_OCCURR_NAME: analyzer.pythia_co_occurrence,
        NGRAM_NAME: analyzer.pythia_relative_ngram,
    }
    plot_average_cross_entropies_over_checkpoints(analyzer, pythia_distributions, "pythia", PYTHIA_CHECKPOINTS, start_year, end_year)
    
    # Plot cross-entropy per year over checkpoints (individual year lines)
    plot_cross_entropies_per_year_over_checkpoints(analyzer, analyzer.olmo_predictions, "olmo", OLMO_CHECKPOINTS, start_year, end_year)
    plot_cross_entropies_per_year_over_checkpoints(analyzer, analyzer.pythia_predictions, "pythia", PYTHIA_CHECKPOINTS, start_year, end_year)
 
def big_bar_plot_predictions_training_data(analyzer):
    cp = 10000 
    year_start = 1000
    year_end = 3000

    analyzer.bar_plot(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, cp, year_start, year_end)
    analyzer.bar_plot(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, cp, year_start, year_end)
    analyzer.bar_plot(analyzer.olmo_co_occurrence, "olmo", CO_OCCURR_NAME, cp, year_start, year_end)
    analyzer.bar_plot(analyzer.olmo_exact_string_match, "olmo", EXACT_STRING_MATCH_NAME, cp, year_start, year_end)
    analyzer.bar_plot(analyzer.pythia_co_occurrence, "pythia", CO_OCCURR_NAME, cp, year_start, year_end)
    analyzer.bar_plot(analyzer.pythia_exact_string_match, "pythia", EXACT_STRING_MATCH_NAME, cp, year_start, year_end)
    analyzer.bar_plot(analyzer.olmo_relative_ngram, "olmo", NGRAM_NAME, cp, year_start, year_end)
    analyzer.bar_plot(analyzer.pythia_relative_ngram, "pythia", NGRAM_NAME, cp, year_start, year_end)

def year_squared_prompts(analyzer):
   # CE between gold dist and year squared prompts
    models_for_prompts = ["allenai_OLMo-2-0425-1B", "EleutherAI_pythia-1.4b-deduped"]
    ce_over_system_prompt_years_olmo = []
    ce_over_system_prompt_years_pythia = []
    prompt_list = []
    for year in range(1950, 2050 + 1):
        prompt = f"The_current_date_is_{year}._In__year__there"
        prompt_list.append(prompt)
        prompt_preds = analyzer.load_other_prompts([prompt], models_for_prompts, all_verbs=False, wordboundary=False, stored_on_disk=True)
        
        ce_for_system_prompt_year_olmo = analyzer.compute_cross_entropy_over_range(prompt_preds[prompt]["allenai_OLMo-2-0425-1B"], "olmo", "final", 1950, 2050, gold_dist_year=year)
        ce_for_system_prompt_year_pythia = analyzer.compute_cross_entropy_over_range(prompt_preds[prompt]["EleutherAI_pythia-1.4b-deduped"], "pythia", "final", 1950, 2050)
        print(f"Calculated CE for {prompt} with OLMo: {ce_for_system_prompt_year_olmo['average_loss']} and Pythia: {ce_for_system_prompt_year_pythia['average_loss']}")
        ce_over_system_prompt_years_olmo.append(ce_for_system_prompt_year_olmo['average_loss'])
        ce_over_system_prompt_years_pythia.append(ce_for_system_prompt_year_pythia['average_loss'])
    analyzer.plot_average_cross_entropies_across_year_prompts(ce_over_system_prompt_years_olmo, ce_over_system_prompt_years_pythia, 1950, 2050, "The current date is [system_prompt_year]. In [year] there")

    


def try_different_system_prompts(analyzer):
    # OLMO INSTRUCT TUNED
    system_years = [2004, 2054]
    basic_prompt_years = [1924, 2124]
    save_paths = []
    ce_per_file = {}
    for base_model, instruct_model in [("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"), ("allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-Instruct"), ("allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-Instruct")]:
        for year in system_years:
            prompts_for_instruct = [f"system_The_current_date_is_{year}.In__year__there", f"system_Current_date:_{year}In__year__there", f"system_Today_Date:_{year}In__year__there", f"system_The_year_is_{year}.In__year__there", f"system_It_is_{year}.In__year__there"]
            prompts_for_base = [f"The_current_date_is_{year}._In__year__there", f"Current_date:_{year}_In__year__there", f"Today_Date:_{year}_In__year__there", f"The_year_is_{year}._In__year__there", f"It_is_{year}._In__year__there"]
                
            base_model = base_model.replace("/", "_")
            instruct_model = instruct_model.replace("/", "_")
            model_name = base_model.split("/")[-1].lower()

            # INSTRUCT
            for prompt in prompts_for_instruct:
                prompt_preds = analyzer.load_other_prompts([prompt], [instruct_model], all_verbs=False, wordboundary=False, stored_on_disk=True)
                data_type_with_prompt = f"{NEXT_TOKEN_NAME}\n{prompt}"
                save_path = analyzer.bar_plot(prompt_preds[prompt][instruct_model], model_name, data_type_with_prompt, "final_instruct", basic_prompt_years[0], basic_prompt_years[1], make_relative=False, system_year=year, folder_name_proposal="instruct_vs_base")
                save_paths.append(save_path)
                ce = analyzer.compute_cross_entropy_over_range(prompt_preds[prompt][instruct_model], instruct_model, "final_instruct", basic_prompt_years[0], basic_prompt_years[1], gold_dist_year=year)
                ce_per_file[save_path] = ce['average_loss']
    
            # BASE
            for prompt in prompts_for_base:
                prompt_preds = analyzer.load_other_prompts([prompt], [base_model], all_verbs=False, wordboundary=False, stored_on_disk=True)
                data_type_with_prompt = f"{NEXT_TOKEN_NAME}\n{prompt}"
                save_path = analyzer.bar_plot(prompt_preds[prompt][base_model], model_name, data_type_with_prompt, "final", basic_prompt_years[0], basic_prompt_years[1], make_relative=False, system_year=year, folder_name_proposal="instruct_vs_base")
                save_paths.append(save_path)
                ce = analyzer.compute_cross_entropy_over_range(prompt_preds[prompt][base_model], base_model, "final", basic_prompt_years[0], basic_prompt_years[1], gold_dist_year=year)
                ce_per_file[save_path] = ce['average_loss']


    # now combine into one plot
    for model_name in ["meta-llama_llama-3.1-8b", "allenai_olmo-2-1124-7b", "allenai_olmo-2-0425-1b"]:
        layout = [
            [f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_The_current_date_is_{system_years[0]}.In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_The_current_date_is_{system_years[0]}._In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_The_current_date_is_{system_years[1]}.In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_The_current_date_is_{system_years[1]}._In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png"],
            [f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_Current_date:_{system_years[0]}In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_Current_date:_{system_years[0]}_In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_Current_date:_{system_years[1]}In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_Current_date:_{system_years[1]}_In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png"],
            [f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_Today_Date:_{system_years[0]}In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_Today_Date:_{system_years[0]}_In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_Today_Date:_{system_years[1]}In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_Today_Date:_{system_years[1]}_In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png"],
            [f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_The_year_is_{system_years[0]}.In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_The_year_is_{system_years[0]}._In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_The_year_is_{system_years[1]}.In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_The_year_is_{system_years[1]}._In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png"],
            [f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_It_is_{system_years[0]}.In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_It_is_{system_years[0]}._In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_instruct_Next-token_predictions_system_It_is_{system_years[1]}.In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png", f"{model_name}_checkpointfinal_Next-token_predictions_It_is_{system_years[1]}._In__year__there_{basic_prompt_years[0]}-{basic_prompt_years[1]}.png"],
        ]
        combine_images_original_sizes(layout, "instruct_vs_base", f"instruct_vs_base/{model_name}.png")

    # now prints files in order of lowest to highest ce
    for file in sorted(ce_per_file, key=ce_per_file.get):
        print(f"{file}: {ce_per_file[file]}")

  
            


if __name__ == "__main__":
    # python get_paper_results.py

    analyzer = AnalyzerClass()

    # filepath = analyzer.save_all_data_to_file()
    # print(f"Data export completed. File saved: {filepath}")

    # bar_plot_predictions_training_data(analyzer)
    # plot_other_model_predictions(analyzer)
    # ce_over_years(analyzer)
    # bar_plot_multiple_checkpoints(analyzer)
    # ce_over_checkpoints(analyzer)
    # big_bar_plot_predictions_training_data(analyzer)
    # year_squared_prompts(analyzer)
    try_different_system_prompts(analyzer) 

    # models = ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-Instruct"]
    # system_years = [2004, 2054]
    # basic_prompt_years = [1924, 2124]
    # ce_losses = {}

    # for year in range(system_years[0], system_years[1] + 1):
    #     prompts_for_instruct = [f"system_The_current_date_is_{year}.In__year__there", f"system_Current_date:_{year}In__year__there", f"system_Today_Date:_{year}In__year__there", f"system_The_year_is_{year}.In__year__there", f"system_It_is_{year}.In__year__there"]
    #     prompts_for_base = [f"The_current_date_is_{year}._In__year__there", f"Current_date:_{year}_In__year__there", f"Today_Date:_{year}_In__year__there", f"The_year_is_{year}._In__year__there", f"It_is_{year}._In__year__there"]
    #     for model in models:
    #         prompts = prompts_for_instruct if "instruct" in model.lower() else prompts_for_base
    #         for prompt in prompts:
    #             prompt_preds = analyzer.load_other_prompts([prompt], [model], all_verbs=False, wordboundary=False, stored_on_disk=True)
    #             ce_loss = analyzer.compute_cross_entropy_over_range(prompt_preds[prompt][model], model, "final", basic_prompt_years[0], basic_prompt_years[1], gold_dist_year=year)
    #             if not f"{model}|{prompt}" in ce_losses:
    #                 ce_losses[f"{model}|{prompt}"] = []
    #             ce_losses[f"{model}|{prompt}"].append(ce_loss['average_loss'])
    # analyzer.plot_average_cross_entropies_across_year_prompts(ce_losses, basic_prompt_years[0], basic_prompt_years[1])








        




    


    


    