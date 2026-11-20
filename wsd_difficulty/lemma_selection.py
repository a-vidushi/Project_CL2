from collections import Counter, defaultdict

from . import config


def group_by_target_word(instances):
    groups = defaultdict(list)
    for instance in instances:
        groups[instance.target_word].append(instance)
    return dict(groups)


def is_eligible(instances,
                minimum_instances=config.MINIMUM_INSTANCES,
                minimum_frequent_senses=config.MINIMUM_FREQUENT_SENSES,
                minimum_instances_per_sense=config.MINIMUM_INSTANCES_PER_SENSE):
    if len(instances) < minimum_instances:
        return False
    sense_counts = Counter(instance.sense for instance in instances)
    frequent_senses = [sense for sense, count in sense_counts.items() if count >= minimum_instances_per_sense]
    return len(frequent_senses) >= minimum_frequent_senses


def select_target_words(instances, **thresholds):
    groups = group_by_target_word(instances)
    return sorted(word for word, word_instances in groups.items() if is_eligible(word_instances, **thresholds))
