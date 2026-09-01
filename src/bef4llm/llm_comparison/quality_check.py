import os
from statistics import mean

import pandas as pd
from tqdm import tqdm

from bef4llm.definitions import (
    Folder,
    Pragmatic_Metrics,
    Pragmatic_Subgroups,
    Similarity_Groups,
    Similarity_Metrics,
    Sytax_Mistakes,
)
from bef4llm.llm_comparison import prepare_datasets
from bef4llm.pragmatic_quality.pragmatic_quality_check import PragmaticQualityCheckBPMN
from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel
from bef4llm.process_models.importer.bpmn_importer import (
    load_bpmn_from_directory,
    load_diagram_from_xml,
)
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.semantic_quality.semantic_quality_check import SemanticQualityCheckBPMN
from bef4llm.synactic_quality.synactic_quality_check import SyntacticQualityCheckBPMN
from bef4llm.validation.validation import validate_bpmn, validate_bpmn_with_error_message


def make_syntactic_check(models, not_valid=None, analyse_mode="detail"):
    """Computes the syntactic quality check for a given set of models.

    Args:
        models: List of models to check, or a directory where models are located.
        not_valid: List of models that should not be checked (path or object).
        analyse_mode: Granularity of the syntactic quality computation
            ("detail", "quality_group_score", "metric_score").

    Returns:
        A list of syntactic quality scores per model (for "detail"/"quality_group_score"),
        or a dict of per-metric score lists (for "metric_score"), plus the list of models
        that could not be analysed.
    """
    if not not_valid:
        not_valid = []

    not_analysed = []
    score_list_syn = []
    score_metric_dict = {Sytax_Mistakes[metric].value: [] for metric in Sytax_Mistakes.__members__}

    if not isinstance(models, list) and os.path.isdir(models):
        model_dir = models
        models = [os.path.join(model_dir, file) for file in os.listdir(model_dir)]

    if len(models) > 0 and isinstance(models[0], str) and os.path.isfile(models[0]):
        new_models = []
        for model in models:
            if model not in not_valid and model.endswith(".bpmn") and model not in not_valid:
                try:
                    new_models.append(load_diagram_from_xml(model))
                except:
                    not_analysed.append(model)
            else:
                not_analysed.append(model)

        models = new_models.copy()

    for model in models:
        try:
            if model.process_graph.number_of_nodes() > 1 and model not in not_valid:
                syn_check = SyntacticQualityCheckBPMN(model)
                if analyse_mode == "detail" or analyse_mode == "quality_group_score":
                    score = syn_check.syntax_check()
                    score_list_syn.append(score)
                elif analyse_mode == "metric_score":
                    metric_score = syn_check.syntax_check_metric_results()
                    for metric in score_metric_dict:
                        score_metric_dict[metric].append(metric_score[metric])

        except Exception as e:
            print("[DEBUG]", e)
            not_analysed.append(model)

    if analyse_mode == "detail" or analyse_mode == "quality_group_score":
        return score_list_syn, not_analysed
    elif analyse_mode == "metric_score":
        return score_metric_dict, not_analysed


def make_semantic_check(models, reference_model, lang, not_valid=None, analyse_mode="detail"):
    """Computes the semantic quality for a given set of models.

    Args:
        models: List of models to check, or a directory where models are located.
        reference_model: Ground-truth model(s) to compare against.
        lang: Language of the textual description.
        not_valid: List of models that should not be checked (path or object).
        analyse_mode: Granularity of the semantic quality computation
            ("detail", "quality_group_score", "metric_score").

    Returns:
        A list of semantic quality scores (for "quality_group_score"), a dict of per-group
        scores (for "detail"), or a dict of per-metric scores (for "metric_score"), plus the
        list of models that could not be analysed.
    """
    if not not_valid:
        not_valid = []

    if not isinstance(models, list) and os.path.isdir(models):
        model_dir = models
        models = [os.path.join(model_dir, file) for file in os.listdir(model_dir)]

    not_analysed = []
    score_list_sem = []
    score_metric_dict = {Similarity_Metrics[metric].value: [] for metric in Similarity_Metrics.__members__}
    score_group_dict = {Similarity_Groups[group].value: [] for group in Similarity_Groups.__members__}

    for model in models:
        try:
            if model not in not_valid and model.endswith(".bpmn"):
                if analyse_mode == "quality_group_score":
                    score = semantic_check_single_model(
                        model, reference_model, lang, analyse_mode=analyse_mode,
                    )
                    if score:
                        score_list_sem.append(score)
                    else:
                        not_analysed.append(model)

                elif analyse_mode == "detail":
                    score_groups = semantic_check_single_model(
                        model, reference_model, lang, analyse_mode=analyse_mode,
                    )
                    if score_groups:
                        for group in score_groups:
                            score_group_dict[group].append(score_groups[group])
                    else:
                        not_analysed.append(model)

                elif analyse_mode == "metric_score":
                    score_metrics = semantic_check_single_model(
                        model, reference_model, lang, analyse_mode=analyse_mode,
                    )
                    if score_metrics:
                        for metric in score_metric_dict:
                            score_metric_dict[metric].append(score_metrics[metric])
                    else:
                        not_analysed.append(model)

        except Exception as e:
            print("[DEBUG]", e)
            not_analysed.append(model)

    if analyse_mode == "quality_group_score":
        return score_list_sem, not_analysed
    elif analyse_mode == "metric_score":
        return score_metric_dict, not_analysed
    elif analyse_mode == "detail":
        return score_group_dict, not_analysed


def semantic_check_single_model(model, reference_model, lang, analyse_mode="detail"):
    """Computes the semantic quality for a single model against one or more ground truths.

    Args:
        model: Model to check, as a file path or CollaborationModel.
        reference_model: Ground-truth model(s) to compare against, as a path, list of paths,
            or (list of) CollaborationModel.
        lang: Language of the textual description.
        analyse_mode: Granularity of the semantic quality computation
            ("detail", "quality_group_score", "metric_score").

    Returns:
        The semantic quality score (for "quality_group_score"), a dict of per-group scores
        (for "detail"), or a dict of per-metric scores (for "metric_score"). Returns None if
        the model can't be scored.
    """
    score_metric_dict = {Similarity_Metrics[metric].value: [] for metric in Similarity_Metrics.__members__}
    score_group_dict = {Similarity_Groups[group].value: [] for group in Similarity_Groups.__members__}
    try:
        if isinstance(model, str):
            bpmn = load_diagram_from_xml(model)
        else:
            bpmn = model

        if not bpmn.process_graph.number_of_nodes() > 1:
            return None

        if isinstance(reference_model, list) and isinstance(reference_model[0], str):
            reference_model = [load_diagram_from_xml(model) for model in reference_model]
        elif isinstance(reference_model, str):
            reference_model = [load_diagram_from_xml(reference_model)]

        if isinstance(reference_model, CollaborationModel):
            sem_check = SemanticQualityCheckBPMN(
                model=bpmn, reference_model=reference_model, lang=lang,
            )
            if analyse_mode == "quality_group_score":
                return sem_check.semantic_quality_check()
            elif analyse_mode == "metric_score":
                return score_metric_dict.update(
                    sem_check.semantic_quality_check_metric_results(),
                )
            elif analyse_mode == "detail":
                return score_group_dict.update(
                    sem_check.semantic_quality_check_detailed(),
                )

        elif isinstance(reference_model, list):
            score_list_sem = []
            for ref_model in reference_model:
                sem_check = SemanticQualityCheckBPMN(
                    model=bpmn, reference_model=ref_model, lang=lang,
                )
                if analyse_mode == "quality_group_score":
                    score = sem_check.semantic_quality_check()
                    score_list_sem.append(score)
                if analyse_mode == "detail":
                    group_scores = sem_check.semantic_quality_check_detailed()
                    for group in group_scores:
                        score_group_dict[group].append(group_scores[group])
                if analyse_mode == "metric_score":
                    metric_score = sem_check.semantic_quality_check_metric_results()
                    for metric in score_metric_dict:
                        score_metric_dict[metric].append(metric_score[metric])

            if analyse_mode == "quality_group_score":
                return mean(score_list_sem)
            elif analyse_mode == "metric_score":
                for group in score_metric_dict:
                    score_metric_dict[group] = mean(score_metric_dict[group])
                return score_metric_dict
            elif analyse_mode == "detail":
                for group in score_group_dict:
                    score_group_dict[group] = mean(score_group_dict[group])
                return score_group_dict
    except Exception as e:
        print("[DEBUG]", e)
        return None


def make_pragmatic_check(models, not_valid=None, analyse_mode="detail"):
    """Computes the pragmatic quality for a given set of models.

    Args:
        models: List of models to check, or a directory where models are located.
        not_valid: List of models that should not be checked (path or object).
        analyse_mode: Granularity of the pragmatic quality computation
            ("detail", "quality_group_score", "metric_score").

    Returns:
        A list of pragmatic quality scores (for "quality_group_score"), a dict of per-group
        scores (for "detail"), or a dict of per-metric scores (for "metric_score"), plus the
        list of models that could not be analysed.
    """
    not_analysed = []
    if not not_valid:
        not_valid = []

    if not isinstance(models, list) and os.path.isdir(models):
        model_dir = models
        models = [os.path.join(model_dir, file) for file in os.listdir(model_dir)]

    if len(models) > 0 and isinstance(models[0], str) and os.path.isfile(models[0]):
        new_models = []
        for model in models:
            if model not in not_valid and model.endswith(".bpmn") and model not in not_valid:
                try:
                    new_models.append(load_diagram_from_xml(model))
                except Exception as e:
                    not_analysed.append(model)
            else:
                not_analysed.append(model)

        models = new_models.copy()

    score_list_prag = []
    score_metric_dict = {Pragmatic_Metrics[metric].value: [] for metric in Pragmatic_Metrics.__members__}
    score_group_dict = {Pragmatic_Subgroups[group].value: [] for group in Pragmatic_Subgroups.__members__}
    for model in models:
        if model not in not_valid:
            try:
                if model.process_graph.number_of_nodes() > 1:
                    prag_check = PragmaticQualityCheckBPMN(model)
                    if analyse_mode == "quality_group_score":
                        score = prag_check.pragmatic_quality_check()
                        score_list_prag.append(score)
                    elif analyse_mode == "metric_score":
                        metric_score = prag_check.pragmatic_quality_check_metric_results()
                        for metric in score_metric_dict:
                            score_metric_dict[metric].append(metric_score[metric])
                    elif analyse_mode == "detail":
                        group_score = prag_check.pragmatic_quality_check_detailed()
                        for metric in score_group_dict:
                            score_group_dict[metric].append(group_score[metric])
                else:
                    not_analysed.append(model)

            except Exception as e:
                print("[DEBUG]", e)
                not_analysed.append(model)

    if analyse_mode == "quality_group_score":
        return score_list_prag, not_analysed
    elif analyse_mode == "metric_score":
        return score_metric_dict, not_analysed
    elif analyse_mode == "detail":
        return score_group_dict, not_analysed


def compute_overall_quality_llms(datasets, test_llm_dir, analyse_method="detaiil", target_file=None, is_human=False):
    """Computes overall quality scores for every LLM found in test_llm_dir.

    Args:
        datasets: Dict of datasets used to check which models were expected per LLM.
        test_llm_dir: Directory containing one subdirectory per LLM being evaluated.
        analyse_method: Granularity of the quality computation
            ("detail", "quality_group_score", "metric_score").
        target_file: CSV file path the resulting dataframe is written to.
        is_human: Whether this run is scoring the human-expert baseline; adds a MEAN row
            over all numeric columns when True.

    Returns:
        pandas.DataFrame: One row per LLM with its aggregated quality scores.
    """
    quality_scores = init_quality_scores_dict(analyse_method, overall_quality_dict=True)

    for llm in tqdm(os.listdir(test_llm_dir)):
        all_invalid_reasons = {}

        if llm.startswith("."):
            continue

        llm_path = os.path.join(test_llm_dir, llm)
        if not os.path.isdir(llm_path):
            continue

        quality_scores["llm"].append(llm)
        quality_scores_llm = init_quality_scores_dict(analyse_method)

        print("[INFO]", llm)

        num_valid = 0
        num_invalid = 0
        num_not_generated = 0
        for dataset in os.listdir(os.path.join(test_llm_dir, llm)):
            if dataset.startswith(".") or os.path.isfile(os.path.join(test_llm_dir, llm, dataset)):
                continue

            model_dir = os.path.join(test_llm_dir, llm, dataset)

            not_valid_xml, invalid_reasons, not_generated = get_invalid_models(
                model_dir, expected_names=datasets[dataset].keys(),
            )
            all_invalid_reasons.update(invalid_reasons)
            num_invalid += len(not_valid_xml)
            num_not_generated += len(not_generated)

            prev_scored = len(quality_scores_llm.get("syntactic quality", []))
            quality_scores_llm, na = get_quality_per_dataset(
                quality_scores_dict=quality_scores_llm,
                models=model_dir,
                analyse_method=analyse_method,
                dataset=datasets[dataset],
                not_valid_xml=not_valid_xml,
            )
            new_scored = len(quality_scores_llm.get("syntactic quality", []))
            num_valid += new_scored - prev_scored

        for key in quality_scores_llm:
            if len(quality_scores_llm[key]) != 0:
                if key in quality_scores:
                    quality_scores[key].append(mean(quality_scores_llm[key]))
                else:
                    quality_scores[key] = [mean(quality_scores_llm[key])]
            else:
                quality_scores[key].append(None)

        num_not_valid = num_invalid + num_not_generated
        total = num_valid + num_not_valid

        quality_scores["validity"].append(num_valid / total if total > 0 else None)
        quality_scores.setdefault("num_valid", []).append(num_valid)
        quality_scores.setdefault("invalid", []).append(num_invalid)
        quality_scores.setdefault("not_generated", []).append(num_not_generated)
        quality_scores.setdefault("not_valid", []).append(num_not_valid)
        quality_scores.setdefault("total", []).append(total)

        if all_invalid_reasons:
            log_path = os.path.join(test_llm_dir, llm, "invalid_models.log")
            with open(log_path, "w") as f:
                for path, reason in all_invalid_reasons.items():
                    f.write(f"FILE: {path}\n")
                    f.write(f"REASON: {reason}\n")
                    f.write("-" * 60 + "\n")
            print(f"[DEBUG] Invalid models log saved at {log_path}")

    df = pd.DataFrame(quality_scores)
    if analyse_method == "quality_group_score":
        df = df.loc[
            :,
            [
                "llm",
                "syntactic quality",
                "pragmatic quality",
                "semantic quality",
                "validity",
                "num_valid",
                "invalid",
                "not_generated",
                "not_valid",
                "total",
            ],
        ]

    if is_human:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        mean_row = df[numeric_cols].mean()
        mean_row["llm"] = "MEAN"
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    df.to_csv(target_file, sep=";")
    return df


def get_quality_per_dataset(quality_scores_dict, analyse_method, dataset, not_valid_xml, models):
    """Scores one dataset's models and merges the results into quality_scores_dict.

    Args:
        quality_scores_dict: Dict accumulating scores across datasets for one LLM.
        analyse_method: Granularity of the quality computation
            ("detail", "quality_group_score", "metric_score").
        dataset: Dict mapping model name to its (language, ..., reference model) info.
        not_valid_xml: List of model paths already known to be invalid.
        models: Directory (or list) of models belonging to this dataset.

    Returns:
        tuple: (quality_scores_dict, not_analysed_xml) — the updated scores dict and the
        list of models that could not be analysed.
    """
    not_analysed_xml = []

    print(f"[DEBUG] dataset keys: {list(dataset.keys())}")
    print(f"[DEBUG] not_valid_xml: {not_valid_xml}")

    syn_scores, na = make_syntactic_check(
        models=models, not_valid=not_valid_xml, analyse_mode=analyse_method,
    )
    print(f"[DEBUG] syn_scores: {syn_scores}")
    not_analysed_xml.extend(na)

    prag_scores, na = make_pragmatic_check(
        models=models, not_valid=not_valid_xml, analyse_mode=analyse_method,
    )
    print(f"[DEBUG] prag_scores: {prag_scores}")
    not_analysed_xml.extend(na)

    if analyse_method == "quality_group_score":
        quality_scores_dict["syntactic quality"].extend(syn_scores)
        quality_scores_dict["pragmatic quality"].extend(prag_scores)

        if isinstance(models, str):
            models = [os.path.join(models, model) for model in os.listdir(models)]

        print(f"[DEBUG] models to evaluate: {models}")

        for model in models:
            if isinstance(model, str):
                model_name = model.split("/")[-1].split(".")[0]
            else:
                model_name = model.name

            print(f"[DEBUG] processing model_name: {model_name}")
            print(f"[DEBUG] model in not_valid_xml: {model in not_valid_xml}")
            print(f"[DEBUG] model_name in dataset: {model_name in dataset}")

            if model not in not_valid_xml:
                if model_name not in dataset:
                    print(f"[WARNING] model_name '{model_name}' not found in dataset, skipping")
                    continue
                semantic_score = semantic_check_single_model(
                    model=model,
                    reference_model=dataset[model_name][2],
                    lang=dataset[model_name][0],
                    analyse_mode=analyse_method,
                )
                print(f"[DEBUG] semantic_score for {model_name}: {semantic_score}")
                if semantic_score:
                    quality_scores_dict["semantic quality"].append(semantic_score)

    # ... rest unchanged
    return quality_scores_dict, not_analysed_xml


def get_invalid_models(model_dir, expected_names=None):
    """Validates every .bpmn file in model_dir and reports missing expected files.

    Args:
        model_dir: Directory containing the .bpmn files to validate.
        expected_names: Optional set of model names (without extension) that should be
            present in model_dir; any missing ones are reported as not generated.

    Returns:
        tuple: (not_valid_xml, invalid_reasons, not_generated) — paths that failed
        validation, a dict mapping each problem path to its reason, and paths for
        expected-but-missing models.
    """
    not_valid_xml = []
    invalid_reasons = {}
    not_generated = []
    found_names = set()

    for model in os.listdir(model_dir):
        if model.endswith(".bpmn"):
            found_names.add(model.split(".")[0])
            path = os.path.join(model_dir, model)
            with open(path) as f:
                content = f.read()
            if content and content != "":
                if not validate_bpmn(path):
                    not_valid_xml.append(path)
                    error_msg, _ = validate_bpmn_with_error_message(path)
                    invalid_reasons[path] = f"[INVALID] {error_msg}"
            else:
                not_valid_xml.append(path)
                invalid_reasons[path] = "[INVALID] Empty file"

    if expected_names is not None:
        missing = set(expected_names) - found_names
        for name in sorted(missing):
            missing_path = os.path.join(model_dir, f"{name}.bpmn")
            not_generated.append(missing_path)
            invalid_reasons[missing_path] = "[NOT_GENERATED] File missing"

    return not_valid_xml, invalid_reasons, not_generated


def init_quality_scores_dict(analyse_method, overall_quality_dict=False):
    """Initializes the dict used to accumulate quality scores.

    Args:
        analyse_method: Granularity of the quality computation
            ("detail", "quality_group_score", "metric_score").
        overall_quality_dict: Whether this dict aggregates multiple LLMs (True) or scores
            a single LLM (False).

    Returns:
        dict: Empty score-list entries keyed by quality metric/group, per analyse_method.
    """
    if analyse_method == "quality_group_score":
        quality_scores = {"syntactic quality": [], "pragmatic quality": [], "semantic quality": []}
    elif analyse_method == "metric_score":
        quality_scores = {Sytax_Mistakes[metric].value: [] for metric in Sytax_Mistakes.__members__}
        quality_scores.update(
            {Pragmatic_Metrics[metric].value: [] for metric in Pragmatic_Metrics.__members__},
        )
        quality_scores.update(
            {Similarity_Metrics[metric].value: [] for metric in Similarity_Metrics.__members__},
        )

    elif analyse_method == "detail":
        quality_scores = {"syntactic quality": []}
        quality_scores.update(
            {Pragmatic_Subgroups[metric].value: [] for metric in Pragmatic_Subgroups.__members__},
        )
        quality_scores.update(
            {Similarity_Groups[metric].value: [] for metric in Similarity_Groups.__members__},
        )
    else:
        quality_scores = dict()

    if overall_quality_dict:
        quality_scores["llm"] = []
        quality_scores["validity"] = []

    return quality_scores


def get_metric_results_per_process_model(datasets, llm_dir, analyse_method, run, target_file=None):
    """Builds a per-process-model results dataframe for one LLM run.

    Returns one row per EXPECTED process model (i.e. every exercise defined in `datasets`,
    regardless of whether it was generated or is valid). Each row carries a "status" column
    ("valid", "invalid", "not_generated") plus the quality scores, which are None whenever
    the model isn't valid or a specific metric couldn't be computed.

    Args:
        datasets: Dict with all datasets (dataset_name -> {exercise_name: (...)}).
        llm_dir: Directory where the process models modelled by this LLM/run are stored.
        analyse_method: Granularity of the quality scores
            ("detail", "quality_group_score", "metric_score").
        run: Identifier of the run being processed.
        target_file: File path the result is saved to (always overwritten, never appended).

    Returns:
        pandas.DataFrame: One row per expected BPMN, with metrics and status as columns.
    """
    def _append_none_scores(scores_dict):
        """Appends None to every metric column except the identifying columns."""
        for metric in scores_dict:
            if metric not in ["process model", "run", "status"]:
                scores_dict[metric].append(None)

    quality_scores = init_quality_scores_dict(analyse_method, overall_quality_dict=False)
    quality_scores["process model"] = []
    quality_scores["run"] = []
    quality_scores["status"] = []

    metric_errors = {}  # Path -> reason, feeds the invalid_models.log

    for dataset in datasets.keys():
        ds_dir = os.path.join(llm_dir, dataset)
        expected_names = datasets[dataset].keys()

        if not os.path.isdir(ds_dir):
            # Whole dataset folder missing -> every exercise in it is not_generated
            for name in expected_names:
                fake_path = os.path.join(ds_dir, f"{name}.bpmn")
                metric_errors[fake_path] = "[NOT_GENERATED] Dataset folder missing"
                quality_scores["process model"].append(name)
                quality_scores["run"].append(f"{run}")
                quality_scores["status"].append("not_generated")
                _append_none_scores(quality_scores)
            continue

        not_valid_xml, invalid_reasons, not_generated = get_invalid_models(
            ds_dir, expected_names=expected_names,
        )
        metric_errors.update(invalid_reasons)

        not_valid_names = {os.path.basename(p).split(".")[0] for p in not_valid_xml}
        not_generated_names = {os.path.basename(p).split(".")[0] for p in not_generated}

        for model_name in sorted(expected_names):
            quality_scores["process model"].append(model_name)
            quality_scores["run"].append(f"{run}")

            if model_name in not_generated_names:
                quality_scores["status"].append("not_generated")
                _append_none_scores(quality_scores)
                continue

            if model_name in not_valid_names:
                quality_scores["status"].append("invalid")
                _append_none_scores(quality_scores)
                continue

            # Valid: actually load and score it
            full_path = os.path.join(ds_dir, f"{model_name}.bpmn")
            quality_scores["status"].append("valid")
            scores = dict()

            try:
                model = load_diagram_from_xml(full_path)
            except Exception as e:
                # Was schema-valid but failed to load as a diagram object -> treat as invalid
                quality_scores["status"][-1] = "invalid"
                metric_errors[full_path] = f"[INVALID] Failed to load BPMN: {e}"
                for metric in quality_scores:
                    if metric not in ["process model", "run", "status"]:
                        scores[metric] = None
                for score in scores:
                    quality_scores[score].append(scores[score])
                continue

            if analyse_method == "metric_score":
                try:
                    syn_check = SyntacticQualityCheckBPMN(model)
                    scores.update(syn_check.syntax_check_metric_results())
                except Exception as e:
                    metric_errors[full_path] = f"[METRIC_ERROR] syntactic: {e}"

                try:
                    prag_check = PragmaticQualityCheckBPMN(model)
                    scores.update(prag_check.pragmatic_quality_check_metric_results())
                except Exception as e:
                    metric_errors[full_path] = f"[METRIC_ERROR] pragmatic: {e}"

                semantic_result = semantic_check_single_model(
                    model=model,
                    reference_model=datasets[dataset][model_name][2],
                    lang=datasets[dataset][model_name][0],
                    analyse_mode=analyse_method,
                )
                if semantic_result is None:
                    metric_errors[full_path] = "[METRIC_ERROR] semantic: returned None"
                    for key in quality_scores:
                        if "semantic" in key:
                            scores[key] = None
                else:
                    scores.update(semantic_result)

            elif analyse_method == "quality_group_score":
                try:
                    syn_check = SyntacticQualityCheckBPMN(model)
                    scores["syntactic quality"] = syn_check.syntax_check()
                except Exception as e:
                    scores["syntactic quality"] = None
                    metric_errors[full_path] = f"[METRIC_ERROR] syntactic: {e}"

                try:
                    prag_check = PragmaticQualityCheckBPMN(model)
                    scores["pragmatic quality"] = prag_check.pragmatic_quality_check()
                except Exception as e:
                    scores["pragmatic quality"] = None
                    metric_errors[full_path] = f"[METRIC_ERROR] pragmatic: {e}"

                try:
                    semantic_score = semantic_check_single_model(
                        model=model,
                        reference_model=datasets[dataset][model_name][2],
                        lang=datasets[dataset][model_name][0],
                        analyse_mode="quality_group_score",
                    )
                    scores["semantic quality"] = semantic_score
                    if semantic_score is None:
                        metric_errors[full_path] = "[METRIC_ERROR] semantic: returned None"
                except Exception as e:
                    scores["semantic quality"] = None
                    metric_errors[full_path] = f"[METRIC_ERROR] semantic: {e}"

            for metric in quality_scores:
                if metric not in ["process model", "run", "status"] and metric not in scores:
                    scores[metric] = None

            for score in scores:
                if score not in quality_scores:
                    quality_scores[score] = []
                quality_scores[score].append(scores[score])

    if metric_errors:
        log_path = os.path.join(llm_dir, "invalid_models.log")
        mode = "a" if os.path.isfile(log_path) else "w"
        with open(log_path, mode) as f:
            for path, reason in metric_errors.items():
                f.write(f"FILE: {path}\n")
                f.write(f"REASON: {reason}\n")
                f.write("-" * 60 + "\n")

    df = pd.DataFrame(quality_scores)

    # Always overwrite: this function already produces the complete dataframe for this
    # llm/run in one pass, so appending would only ever duplicate rows across repeated
    # invocations of the command.
    lead_cols = ["process model", "run", "status"]
    other_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + other_cols]

    df.to_csv(target_file, sep=";", index=False)
    return df


def get_datasets_config():
    """Builds the dataset config dict based on the DATASET_MODE/DATASETS env vars.

    Returns:
        dict: Dataset name -> dataset config, as produced by prepare_datasets.

    Raises:
        RuntimeError: If DATASET_MODE (or DATASETS, when required) is not set.
        ValueError: If DATASET_MODE or a name in DATASETS is invalid.
    """
    mode = os.getenv("DATASET_MODE")
    if not mode:
        raise RuntimeError(
            "DATASET_MODE is not set in .env\n"
            "Options: general | experts",
        )
    mode = mode.strip().lower()

    if mode == "experts":
        print("[DEBUG] Using expert dataset only")
        return {"experts": prepare_datasets.prepare_experts_comparison()}

    elif mode == "general":
        datasets_env = os.getenv("DATASETS")
        if not datasets_env:
            raise RuntimeError("DATASETS must be set when DATASET_MODE=general")

        dataset_names = [d.strip() for d in datasets_env.split(",") if d.strip()]

        fixed_allowed = {"camunda", "bpmn_and_text", "lre_new", "lre_old"}

        def is_valid(name):
            """Checks whether a dataset name is a fixed name or a valid camunda_N variant."""
            if name in fixed_allowed:
                return True
            parts = name.split("_")
            return len(parts) == 2 and parts[0] == "camunda" and parts[1].isdigit()

        invalid = [d for d in dataset_names if not is_valid(d)]
        if invalid:
            raise ValueError(f"Invalid datasets in DATASETS: {invalid}")

        datasets = {}
        for name in dataset_names:
            if name == "camunda":
                datasets[name] = prepare_datasets.prepare_camunda()
            elif name.startswith("camunda_") and name.split("_")[1].isdigit():
                version = int(name.split("_")[1])
                datasets[name] = prepare_datasets.prepare_camunda(version=version)
            elif name == "bpmn_and_text":
                datasets[name] = prepare_datasets.prepare_text_and_bpmn()
            elif name == "lre_new":
                datasets[name] = prepare_datasets.prepare_lre_new()
            elif name == "lre_old":
                datasets[name] = prepare_datasets.prepare_lre_old()

        print(f"[DEBUG] Using datasets: {list(datasets.keys())}")
        return datasets

    else:
        raise ValueError("DATASET_MODE must be 'general' or 'experts'")


def run_quality_check_for_run(run, evaluation):
    """Resolves datasets for a run and computes overall LLM quality for it.

    Includes camunda_N subfolders not present in the default dataset config, then computes
    overall LLM quality for that run via compute_overall_quality_llms. This contains the
    dataset-resolution logic that used to live in compare_llms.py's check_quality_llms,
    moved here so it can be reused by other modules (e.g. table generation) without a
    circular import.

    Args:
        run: Identifier of the run to resolve datasets and compute quality for.
        evaluation: Granularity of the quality computation, passed through as analyse_method.

    Returns:
        pandas.DataFrame: Same dataframe returned by compute_overall_quality_llms.
    """
    datasets = get_datasets_config()

    llm_run_dir = os.path.join(get_folder_path(Folder.DATA), f"llm_run{run}")
    for llm in os.listdir(llm_run_dir):
        llm_path = os.path.join(llm_run_dir, llm)
        if not os.path.isdir(llm_path) or llm.startswith("."):
            continue
        for dataset_dir in os.listdir(llm_path):
            if dataset_dir not in datasets:
                parts = dataset_dir.split("_")
                if len(parts) == 2 and parts[0] == "camunda" and parts[1].isdigit():
                    version = int(parts[1])
                    datasets[dataset_dir] = prepare_datasets.prepare_camunda(version=version)

    df = compute_overall_quality_llms(
        datasets=datasets,
        test_llm_dir=llm_run_dir,
        target_file=os.path.join(
            get_folder_path(Folder.DATA), f"llm_run{run}", f"llm_results_run{run}_{evaluation}.csv",
        ),
        analyse_method=evaluation,
    )
    return df