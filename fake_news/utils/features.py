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
    "Tennessee": "Tennessee",
    "Tennesse": "Tennessee",
    "District of Columbia": "Washington D.C.",
    "Washington DC": "Washington D.C.",
    "Washington, D.C.": "Washington D.C.",
    "Washington D.C.": "Washington D.C.",
    "Tex": "Texas",
    "Texas": "Texas",
    "Washington state": "Washington",
    "Washington": "Washington",
    "Virgina": "Virginia",
    "Virgiia": "Virginia",
    "Virginia": "Virginia",
    "Pennsylvania": "Pennsylvania",
    "PA - Pennsylvania": "Pennsylvania",
    "Rhode Island": "Rhode Island",
    "Rhode island": "Rhode Island",
    "Ohio": "Ohio",
    "ohio": "Ohio"
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

BARELY_TRUE_CREDIT_BINS = [0., 3.33333333, 6.66666667, 10., 13.33333333,
                           16.66666667, 20., 23.33333333, 26.66666667, 30.,
                           33.33333333, 36.66666667, 40., 43.33333333, 46.66666667,
                           50., 53.33333333, 56.66666667, 60., 63.33333333,
                           66.66666667, 70.]

FALSE_CREDIT_BINS = [0., 5.18181818, 10.36363636, 15.54545455,
                     20.72727273, 25.90909091, 31.09090909, 36.27272727,
                     41.45454545, 46.63636364, 51.81818182, 57.,
                     62.18181818, 67.36363636, 72.54545455, 77.72727273,
                     82.90909091, 88.09090909, 93.27272727, 98.45454545,
                     103.63636364, 108.81818182, 114.]

HALF_TRUE_BINS = [0., 7.27272727, 14.54545455, 21.81818182,
                  29.09090909, 36.36363636, 43.63636364, 50.90909091,
                  58.18181818, 65.45454545, 72.72727273, 80.,
                  87.27272727, 94.54545455, 101.81818182, 109.09090909,
                  116.36363636, 123.63636364, 130.90909091, 138.18181818,
                  145.45454545, 152.72727273, 160.]

MOSTLY_TRUE_BINS = [0., 7.40909091, 14.81818182, 22.22727273,
                    29.63636364, 37.04545455, 44.45454545, 51.86363636,
                    59.27272727, 66.68181818, 74.09090909, 81.5,
                    88.90909091, 96.31818182, 103.72727273, 111.13636364,
                    118.54545455, 125.95454545, 133.36363636, 140.77272727,
                    148.18181818, 155.59090909, 163.]

PANTS_FIRE_BINS = [0., 4.77272727, 9.54545455, 14.31818182,
                   19.09090909, 23.86363636, 28.63636364, 33.40909091,
                   38.18181818, 42.95454545, 47.72727273, 52.5,
                   57.27272727, 62.04545455, 66.81818182, 71.59090909,
                   76.36363636, 81.13636364, 85.90909091, 90.68181818,
                   95.45454545, 100.22727273, 105.]


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
