from bef4llm.benchmark import prepare_datasets
from bef4llm.benchmark.quality_check import *
from bef4llm.datasets.model_collection import DatasetCollection, download_any_format, download_git_folder
from bef4llm.resource_controller.path_helper import look_for_directory
from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language


def get_baseline_model_dirs():
    """
    returns dirs to folders, which are used to compute the baseline, based on the datasets used for benchmark
    """
    model_dirs = get_model_dirs_camunda()
    model_dirs + get_model_dirs_text_and_bpmn()
    model_dirs += get_model_dirs_lre()

    return model_dirs


def get_model_dirs_camunda():
    """
    returns the directories of the camunda datasets (dir for each BPMN XML file)
    """
    def get_model_folder(lang):
        dir_path = look_for_directory("BPMN for Research")
        model_folder = ""
        if lang == Language.ENGLISH:
            dir_path = os.path.join(dir_path, "English")
            model_folder = "03-Solution"
        elif lang == Language.GERMAN:
            dir_path = os.path.join(dir_path, "German")
            model_folder = "03-Musterlösung"

        return dir_path, model_folder

    def extract_model_dirs(lang):
        model_list = []
        dir_path, model_folder = get_model_folder(Language.ENGLISH)
        for exercise in os.listdir(dir_path):
            if not exercise.startswith("."):
                model_dir = os.path.join(dir_path, exercise, model_folder)

                for model in os.listdir(model_dir):
                    model_list.append(model)

        return model_list

    # add dowlooad files if not yet done
    if not look_for_directory("camunda"):
        data = DatasetCollection.CAMUNDA.value
        download_any_format(url=data["link"],
                            file_name=data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), data["destination_dir"]),
                            compressed=True)

    return extract_model_dirs(Language.ENGLISH) + extract_model_dirs(Language.GERMAN)


def get_model_dirs_text_and_bpmn():
    """
    returns the directories of the bpmn and text datasets (dir for each BPMN XML file)
    """

    # add dowlooad files if not yet done
    if not look_for_directory("text_and_bpmn"):
        data = DatasetCollection.TEXT_AND_BPMN.value
        download_any_format(url=data["link"],
                            file_name=data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), data["destination_dir"]),
                            compressed=True)

    model_list = []
    path = f"{get_folder_path(Folder.DATA)}/models/text_and_bpmn/bpmn"
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):

            filename = file.split(".")[0]
            for model in os.listdir(os.path.join(path, filename)):
                # iter over all models
                if model.endswith(".txt"):
                    continue

                model_name = model.split(".")[0]

                # model quality is not documented
                if not os.path.isfile(f"{path}/{filename}/{model_name}.quality.txt"):
                    continue

                # select only models with quality 5
                with open(f"{path}/{filename}/{model_name}.quality.txt", 'r') as f:
                    quality = f.read()

                if quality == "5":
                    model_list.append(model)

    return model_list

def get_model_dirs_lre():
    """
    returns the directories of the lre datasets (dir for each BPMN XML file)
    """
    def extract_model_dirs(lre_folder):
        model_list = []
        camunda = ["Dispatch-of-goods", "Recourse", "Self-service-restaurant"]

        for file in os.listdir(f"{lre_folder}/Texts"):
            try:
                if not file.startswith("."):
                    filename = file.split(".")[0]

                    # some camunda files are added -> no duplicates
                    if filename in camunda and not camunda:
                        continue

                    model_list.append(f"{lre_folder}/Models/{filename}.bpmn")

            except Exception as e:
                print(f"For file {file} no model text_pair was built due to an error: {e}")

        return model_list

    lre_original_data = DatasetCollection.LRE_ORIGINAL.value
    if not look_for_directory("lre_original"):
        download_git_folder(url=lre_original_data["link"],
                            file_name=lre_original_data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), lre_original_data["destination_dir"]),
                            download_folder=lre_original_data["download_folder"],
                            download_dir=lre_original_data["download_dir"],
                            delete_folder=lre_original_data["delete_folder"])

    lre_original_folder = os.path.join(get_folder_path(Folder.DATA), lre_original_data["destination_dir"])

    lre_new_data = DatasetCollection.LRE_NEW.value
    if not look_for_directory("lre_new"):
        download_git_folder(url=lre_new_data["link"],
                            file_name=lre_new_data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), lre_new_data["destination_dir"]),
                            download_folder=lre_new_data["download_folder"],
                            download_dir=lre_new_data["download_dir"],
                            delete_folder=lre_new_data["delete_folder"])

    lre_new_folder = os.path.join(get_folder_path(Folder.DATA), lre_new_data["destination_dir"])

    return extract_model_dirs(lre_new_folder) + extract_model_dirs(lre_original_folder)




def baseline_ground_truth_bpmn(analyse_method="detail"):
    """
    Computes the quality scores for the BPMNs, which are used in the benchmark (only syntactic and pragmatic possible)
    """

    def get_invalid_models(models):
        not_valid_xml = []
        for model in models:
            if not validate_bpmn(model):
                not_valid_xml.append(model)

        return not_valid_xml

    models = get_baseline_model_dirs()
    not_valid_xml = get_invalid_models(models)
    quality_scores = {"not_valid": len(not_valid_xml), "valid": len(models) - len(not_valid_xml), "llm": "baseline_ground_truth_bpmn"}

    syn_scores, _ = make_syntactic_check(models=models,
                                         not_valid=not_valid_xml,
                                         analyse_mode=analyse_method)
    prag_scores, _ = make_pragmatic_check(models=models,
                                          not_valid=not_valid_xml,
                                          analyse_mode=analyse_method)

    if analyse_method == "quality_group_score":
        quality_scores["synatic quality"] = mean(syn_scores)
        quality_scores["pragmatic quality"] = mean(prag_scores)

    else:
        if analyse_method == "detail":
            quality_scores["synatic quality"].extend(syn_scores)
        else:
            for key in syn_scores:
                if key not in quality_scores:
                    quality_scores[key] = []
                quality_scores[key].extend(syn_scores[key])
        for key in prag_scores:
            if key not in quality_scores:
                quality_scores[key] = []
            quality_scores[key].extend(prag_scores[key])

    return quality_scores

def get_baseline_camunda(analyse_method="detail"):
    """
    Computes the quality of the student modelled BPMNs in the camunda dataset

    Parameters
    ----------
    analyse_method : str
        method of analysis (detail, metric_score, quality_group_score)

    Returns
    -------
    quality_scores : dict
        Quality scores for the BPMNs in the camunda dataset (only the student modelled)
    """

    def get_invalid_models(models):
        """
        returns list of invalid BPMN models
        """
        not_valid_xml = []
        for model in models:
            if not validate_bpmn(model):
                not_valid_xml.append(model)

        return not_valid_xml

    def extract_files(lang, model_pairs):
        """
        extracts text models pairs from dataset
        """
        dir_path = look_for_directory("BPMN for Research")
        model_dir = ""
        ground_truth_dir = ""
        if lang == Language.ENGLISH:
            dir_path = os.path.join(dir_path, "English")
            ground_truth_dir = "03-Solution"
            model_dir = "02-Results"
        elif lang == Language.GERMAN:
            dir_path = os.path.join(dir_path, "German")
            ground_truth_dir = "03-Musterlösung"
            model_dir = "02-Ergebnisse"

        for exercise in os.listdir(dir_path):
            # exclude hidden directories
            if not exercise.startswith("."):
                if exercise not in model_pairs:
                    model_pairs[exercise] = dict()
                ground_truth_bpmn_model = os.path.join(dir_path, exercise, ground_truth_dir)
                ground_truth_bpmn_model = [os.path.join(ground_truth_bpmn_model, file) for file in os.listdir(ground_truth_bpmn_model)
                                           if not file.startswith(".")]


                model_pairs[exercise]["ground truth"] = ground_truth_bpmn_model
                model_pairs[exercise]["models"] = []

                for model in os.listdir(os.path.join(dir_path, exercise, model_dir)):
                    if model.endswith(".bpmn"):
                        model_pairs[exercise]["models"].append(os.path.join(dir_path, exercise, model_dir, model))


        return model_pairs

    bpmn_model_pairs = dict()
    bpmn_model_pairs = extract_files(lang=Language.ENGLISH, model_pairs=bpmn_model_pairs)
    bpmn_model_pairs = extract_files(lang=Language.GERMAN, model_pairs=bpmn_model_pairs)


    if analyse_method == "quality_group_score":
        quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline",
                          "syntatic quality": [], "pragmatic quality": [], "semantic quality": []}
    else:
        if analyse_method == "detail":
            quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline", "syntatic quality": []}
        else:
            quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline"}

    for exercise in tqdm(bpmn_model_pairs):
        if not exercise.startswith("."):
            models = bpmn_model_pairs[exercise]["models"]
            ground_truth_bpmn_model = bpmn_model_pairs[exercise]["ground truth"]

            not_valid_xml = get_invalid_models(models)

            syn_scores, not_valid_xml = make_syntactic_check(models=models,
                                                 not_valid=not_valid_xml,
                                                 analyse_mode=analyse_method)
            prag_scores, not_valid_xml = make_pragmatic_check(models=models,
                                                  not_valid=not_valid_xml,
                                                  analyse_mode=analyse_method)
            semantic_score, not_valid_xml = make_semantic_check(
                models=models,
                reference_model=ground_truth_bpmn_model,
                lang=Language.ENGLISH,  # when multiple language implementation, this must be changed
                analyse_mode=analyse_method)


            if analyse_method == "quality_group_score":
                quality_scores["syntatic quality"].extend(syn_scores)
                quality_scores["pragmatic quality"].extend(prag_scores)
                quality_scores["semantic quality"].extend(semantic_score)

            else:
                if analyse_method == "detail":
                    quality_scores["syntatic quality"].extend(syn_scores)
                else:
                    for key in syn_scores:
                        if key not in quality_scores:
                            quality_scores[key] = []
                        quality_scores[key].extend(syn_scores[key])
                for key in prag_scores:
                    if key not in quality_scores:
                        quality_scores[key] = []
                    quality_scores[key].extend(prag_scores[key])

                for key in semantic_score:
                    if key not in quality_scores:
                        quality_scores[key] = []
                    quality_scores[key].extend(semantic_score[key])
            """
            if analyse_method == "quality_group_score":
                quality_scores["synatic quality"] = mean(syn_scores)
                quality_scores["pragmatic quality"] = mean(prag_scores)

            else:
                if analyse_method == "detail":
                    quality_scores["synatic quality"].extend(syn_scores)
                else:
                    for key in syn_scores:
                        if key not in quality_scores:
                            quality_scores[key] = []
                        quality_scores[key].extend(syn_scores[key])
                for key in prag_scores:
                    if key not in quality_scores:
                        quality_scores[key] = []
                    quality_scores[key].extend(prag_scores[key])
            """
            quality_scores["not_valid"] += len(not_valid_xml)
            quality_scores["valid"] += len(models) - len(not_valid_xml)


    for key in quality_scores:
        if isinstance(quality_scores[key], list):
            print(key, quality_scores[key])
            quality_scores[key] = mean(quality_scores[key])

    return quality_scores

def get_expert_scores(analyse_method):
    """
    Computes quality scores for expert dataset

    Parameters
    ----------
        analyse_method: str
            method of analysis (detail, metric_score, quality_group_score)

    Returns
    -------
    quality_scores: dict
        quality scores for expert dataset
    """
    def get_invalid_models(models):
        not_valid_xml = []
        for model in models:
            if not validate_bpmn(model):
                not_valid_xml.append(model)

        return not_valid_xml

    def extract_files(model_pairs):
        dir_path = look_for_directory("Expert_BPMN")
        model_dir = "experts_modelling"
        ground_truth_dir = "ground_truth_bpmn"

        for expert in os.listdir(os.path.join(dir_path, model_dir)):
            # exclude hidden directories
            if not expert.startswith("."):
                for model in os.listdir(os.path.join(dir_path, model_dir, expert)):
                    if model.startswith("."):
                        continue
                    if model not in model_pairs:
                        if "schufa" not in model:
                            ground_truth_bpmn_model = os.path.join(dir_path, ground_truth_dir, model)
                        else:
                            ground_truth_bpmn_model = [os.path.join(dir_path, ground_truth_dir, model.split(".")[0] + "1.bpmn")]
                            ground_truth_bpmn_model.append(os.path.join(dir_path, ground_truth_dir,
                                                                   model.split(".")[0] + "2.bpmn"))

                        model_pairs[model] = dict()
                        model_pairs[model]["ground truth"] = ground_truth_bpmn_model
                        model_pairs[model]["models"] = [os.path.join(dir_path, model_dir, expert, model)]
                    else:
                        model_pairs[model]["models"].append(os.path.join(dir_path, model_dir, expert, model))

        return model_pairs

    bpmn_model_pairs = dict()
    bpmn_model_pairs = extract_files(model_pairs=bpmn_model_pairs)

    if analyse_method == "quality_group_score":
        quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline",
                          "syntatic quality": [], "pragmatic quality": [], "semantic quality": []}
    else:
        if analyse_method == "detail":
            quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline", "syntatic quality": []}
        else:
            quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline"}

    for model in tqdm(bpmn_model_pairs):
        if not model.startswith("."):
            models = bpmn_model_pairs[model]["models"]
            ground_truth_bpmn_model = bpmn_model_pairs[model]["ground truth"]

            not_valid_xml = get_invalid_models(models)

            syn_scores, not_valid_xml = make_syntactic_check(models=models,
                                                 not_valid=not_valid_xml,
                                                 analyse_mode=analyse_method)
            prag_scores, not_valid_xml = make_pragmatic_check(models=models,
                                                  not_valid=not_valid_xml,
                                                  analyse_mode=analyse_method)
            semantic_score, not_valid_xml = make_semantic_check(
                models=models,
                reference_model=ground_truth_bpmn_model,
                lang=Language.ENGLISH,  # when multiple language implementation, this must be changed
                analyse_mode=analyse_method)


            if analyse_method == "quality_group_score":
                quality_scores["syntatic quality"].extend(syn_scores)
                quality_scores["pragmatic quality"].extend(prag_scores)
                quality_scores["semantic quality"].extend(semantic_score)

            else:
                if analyse_method == "detail":
                    quality_scores["syntatic quality"].extend(syn_scores)
                else:
                    for key in syn_scores:
                        if key not in quality_scores:
                            quality_scores[key] = []
                        quality_scores[key].extend(syn_scores[key])
                for key in prag_scores:
                    if key not in quality_scores:
                        quality_scores[key] = []
                    quality_scores[key].extend(prag_scores[key])

                for key in semantic_score:
                    if key not in quality_scores:
                        quality_scores[key] = []
                    quality_scores[key].extend(semantic_score[key])

            quality_scores["not_valid"] += len(not_valid_xml)
            quality_scores["valid"] += len(models) - len(not_valid_xml)


    for key in quality_scores:
        if isinstance(quality_scores[key], list):
            print(key, quality_scores[key])
            quality_scores[key] = mean(quality_scores[key])

    return quality_scores



