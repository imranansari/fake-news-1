from copy import deepcopy
from typing import Dict
from typing import List

# Mapping from speaker title variants seen in data to their canonical form
CANONICAL_SPEAKER_TITLES = {
    "u.s. house of representative": "u.s. house of representatives",
    "u.s. representativej": "u.s. representative",
    "talks show host": "talk show host",
    "u. s. congressman": "u.s. congressman",
    "politican action committee": "political action committee",
    "retired": "retiree",
    "restauranteur": "restaurateur"
}

SIX_WAY_LABEL_TO_BINARY = {
    "pants-fire": False,
    "barely-true": False,
    "false": False,
    "true": True,
    "half-true": True,
    "mostly-true": True
}

CANONICAL_STATE = {
    "tennesse": "tennessee",
    "district of columbia": "washington d.c.",
    "washington dc": "washington d.c.",
    "washington, d.c.": "washington d.c.",
    "washington d.c.": "washington d.c.",
    "tex": "texas",
    "texas": "texas",
    "washington state": "washington",
    "washington": "washington",
    "virgina": "virginia",
    "virgiia": "virginia",
    "virginia": "virginia",
    "pennsylvania": "pennsylvania",
    "pa - pennsylvania": "pennsylvania",
    "rhode island": "rhode island",
    "rhode island": "rhode island",
    "ohio": "ohio",
    "ohio": "ohio"
}

PARTY_AFFILIATIONS = {
    "republican", "democrat", "none", "organization", "independent",
    "columnist", "activist", "talk-show-host", "libertarian",
    "newsmaker", "journalist", "labor-leader", "state-official",
    "business-leader", "education-official", "tea-party-member",
    "green", "liberal-party-canada", "government-body", "Moderate",
    "democratic-farmer-labor", "ocean-state-tea-party-action",
    "constitution-party"
}


def compute_bin_idx(val: float, bins: List[float]) -> int:
    for idx, bin_val in enumerate(bins):
        if val <= bin_val:
            return idx


# NOTE: Making sure that all normalization operations preserve immutability of inputs
def normalize_labels(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        # First do simple cleaning
        normalized_datapoint = deepcopy(datapoint)
        normalized_datapoint["label"] = SIX_WAY_LABEL_TO_BINARY[datapoint["label".lower().strip()]]
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_speaker_title(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        # First do simple cleaning
        normalized_datapoint = deepcopy(datapoint)
        old_speaker_title = normalized_datapoint["speaker_title"]
        old_speaker_title = old_speaker_title.lower().strip().replace("-", " ")
        # Then canonicalize
        if old_speaker_title in CANONICAL_SPEAKER_TITLES:
            old_speaker_title = CANONICAL_SPEAKER_TITLES[old_speaker_title]
        normalized_datapoint["speaker_title"] = old_speaker_title
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_party_affiliations(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        if normalized_datapoint["party_affiliation"] not in PARTY_AFFILIATIONS:
            normalized_datapoint["party_affiliation"] = "none"
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_state_info(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        old_state_info = normalized_datapoint["state_info"]
        old_state_info = old_state_info.lower().strip().replace("-", " ")
        if old_state_info in CANONICAL_STATE:
            old_state_info = CANONICAL_STATE[old_state_info]
        normalized_datapoint["state_info"] = old_state_info
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_counts(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for idx, datapoint in enumerate(datapoints):
        normalized_datapoint = deepcopy(datapoint)
        for count_col in ["barely_true_count",
                          "false_count",
                          "half_true_count",
                          "mostly_true_count",
                          "pants_fire_count"]:
            if count_col in normalized_datapoint:
                normalized_datapoint[count_col] = float(normalized_datapoint[count_col])
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean(datapoints: List[Dict]) -> List[Dict]:
    return normalize_and_clean_speaker_title(
        normalize_and_clean_party_affiliations(
            normalize_and_clean_state_info(
                normalize_and_clean_counts(
                    normalize_labels(
                        datapoints
                    )
                )
            )
        )
    )
