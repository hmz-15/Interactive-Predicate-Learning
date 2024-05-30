from predicate_learning.utils.registry import Registry

LEARNER = Registry("Learner")


def create_learner(learner_name: str, *args, **kwargs):
    return LEARNER.get(learner_name)(*args, **kwargs)
