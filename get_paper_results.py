from analyzer_class import (
    AnalyzerClass, 
    plot_cross_entropies_per_year_over_checkpoints, plot_average_cross_entropies_over_checkpoints, plot_cross_entropies,
    CO_OCCURR_NAME, EXACT_STRING_MATCH_NAME, NGRAM_NAME, NEXT_TOKEN_NAME,
    OLMO_CHECKPOINTS, PYTHIA_CHECKPOINTS,
    PROMPT_DISPLAY_NAMES,
)

def bar_plot_predictions_training_data():
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

def plot_other_model_predictions():
    start_year = 1950
    end_year = 2050

    # plot_model_predictions with other models
    for model_name in analyzer.other_model_predictions.keys():
        analyzer.bar_plot(analyzer.other_model_predictions[model_name], model_name, NEXT_TOKEN_NAME, "final", start_year, end_year, make_relative=False, separate_present_future=True)

def plot_other_prompt_predictions():
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

def ce_over_years():
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
    
def bar_plot_multiple_checkpoints():
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
    
def ce_over_checkpoints():
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
 
def big_bar_plot_predictions_training_data():
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

if __name__ == "__main__":
    # python get_paper_results.py

    analyzer = AnalyzerClass()

    # filepath = analyzer.save_all_data_to_file()
    # print(f"Data export completed. File saved: {filepath}")

    # bar_plot_predictions_training_data()
    # plot_other_model_predictions()
    # ce_over_years()
    # bar_plot_multiple_checkpoints()
    # ce_over_checkpoints()
    # big_bar_plot_predictions_training_data()

    cp = 10000
    start_year = 1950
    end_year = 2050


    # year squared prompts
    # models_for_prompts = ["allenai_OLMo-2-0425-1B", "EleutherAI_pythia-1.4b-deduped"]
    # for year in [1950, 2000, 2050]:
    #     prompt = f"The_year_is_{year}._In__year__there"
    #     prompt_preds = analyzer.load_other_prompts([prompt], models_for_prompts, all_verbs=True, wordboundary=False, stored_on_disk=True)
    #     data_type_with_prompt = f"{NEXT_TOKEN_NAME}\nThe year is {year}. In [year] there"
    #     for model_name in models_for_prompts:
    #         # analyzer.bar_plot(prompt_preds[prompt][model_name], model_name, data_type_with_prompt, "final", 1600, 2400, make_relative=False)
    #         analyzer.bar_plot(prompt_preds[prompt][model_name], model_name, data_type_with_prompt, "final", 1900, 2100, make_relative=False)


    analyzer.bar_plot(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, "final", 1600, 2400)
    analyzer.bar_plot(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, "final", 1600, 2400)
    analyzer.bar_plot(analyzer.olmo_predictions, "olmo", NEXT_TOKEN_NAME, "final", 1900, 2100)
    analyzer.bar_plot(analyzer.pythia_predictions, "pythia", NEXT_TOKEN_NAME, "final", 1900, 2100)

    models_for_prompts = ["allenai_OLMo-2-0425-1B", "EleutherAI_pythia-1.4b-deduped"]
    prompt = "In_the_year___year__there"
    prompt_preds = analyzer.load_other_prompts([prompt], models_for_prompts, all_verbs=False, wordboundary=False, stored_on_disk=False)
    data_type_with_prompt = f"{NEXT_TOKEN_NAME}\nIn the year [year] there"
    for model_name in models_for_prompts:
        analyzer.bar_plot(prompt_preds[prompt][model_name], model_name, data_type_with_prompt, "final", 1600, 2400, make_relative=False)
        analyzer.bar_plot(prompt_preds[prompt][model_name], model_name, data_type_with_prompt, "final", 1900, 2100, make_relative=False)
    
        


    