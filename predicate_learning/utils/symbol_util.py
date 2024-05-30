import torch
import logging
import itertools

from pddlgym.structs import Type, Predicate
from .general_util import Timer

############# Symbolic Models ###############

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
t = Timer(logger=logger, color="green")


def aggr_predicate_variables(predicate, node_var_dict, edge_index=None):
    # edge_index (B, num_edges, 2)
    arity = predicate.arity
    entity_types = predicate.var_types

    if arity == 1:
        return node_var_dict[entity_types[0]]

    elif arity == 2:
        assert edge_index is not None
        batch_idx = (
            torch.arange(edge_index.shape[0])
            .unsqueeze(-1)
            .expand(edge_index.shape[0], edge_index.shape[1])
            .reshape(-1)
        )
        x_1 = node_var_dict[entity_types[0]][batch_idx, edge_index[:, :, 0].reshape(-1)].reshape(
            edge_index.shape[0], edge_index.shape[1], -1
        )
        x_2 = node_var_dict[entity_types[1]][batch_idx, edge_index[:, :, 1].reshape(-1)].reshape(
            edge_index.shape[0], edge_index.shape[1], -1
        )

        x = torch.cat([x_1, x_2], dim=-1)
        return x

    else:
        raise NotImplementedError


def get_object_combinations(type_to_objs, var_types, allow_duplicates=False):
    choices = [type_to_objs[vt] for vt in var_types]

    for choice in itertools.product(*choices):
        if not allow_duplicates and len(set(choice)) != len(choice):
            continue
        yield choice


def read_predicates_info_from_obs(
    predicates,
    predicate_to_idx,
    obs,
    return_labels=False,
    return_batch=True,
    return_names=True,
):
    labels_dict = {}
    batch_dict = {}
    names_dict = {}
    for pred_name, predicate in predicates.items():
        info_dict = _read_predicate_info_from_obs(predicate, predicate_to_idx[pred_name], obs)

        if return_labels:
            labels_dict[pred_name] = info_dict["labels"]
        if return_batch:
            batch_dict[pred_name] = info_dict["batch"]
        if return_names:
            names_dict[pred_name] = info_dict["names"]

    out_dict = {}
    if return_labels:
        out_dict["labels_dict"] = labels_dict
    if return_batch:
        out_dict["batch_dict"] = batch_dict
    if return_names:
        out_dict["names_dict"] = names_dict

    return out_dict


def _read_predicate_info_from_obs(predicate, key_idx_pair, obs):
    arity, entity_types = predicate.arity, predicate.var_types
    data_key, idx_in_key = key_idx_pair

    # get batch
    if hasattr(obs[entity_types[0]], "batch"):
        # if data is batched
        batch = obs[entity_types[0]].batch
    else:
        # if single data
        batch_dim = obs[entity_types[0]].x.shape[0]
        batch_device = obs[entity_types[0]].x.device
        batch = torch.zeros(batch_dim, dtype=torch.long, device=batch_device)

    if arity == 2:
        batch = batch[obs[data_key].edge_index[0, :]]

    # get names
    names = None
    if arity == 1:
        names = [[name] for name in obs[data_key].names]

    elif arity == 2:
        edge_index = obs[data_key].edge_index
        names = [
            [
                obs[entity_types[0]].names[edge_index[0, edge_i]],
                obs[entity_types[1]].names[edge_index[1, edge_i]],
            ]
            for edge_i in range(edge_index.shape[1])
        ]

    else:
        raise NotImplementedError

    # get labels
    labels = None
    if hasattr(obs[data_key], "labels"):
        labels = obs[data_key].labels[..., idx_in_key]

    return {"labels": labels, "names": names, "batch": batch}


def read_entities_list(names_dict, batch_dict, predicates, **kwargs):
    t.tic("read entities list")
    example_pred_name = list(predicates.keys())[0]
    batch_size = batch_dict[example_pred_name][-1].item() + 1

    entities_list = [set() for _ in range(batch_size)]

    for pred_name in predicates.keys():
        predicate = predicates[pred_name]
        names, batch = (
            names_dict[pred_name],
            batch_dict[pred_name],
        )

        for names_per_lit, batch_id in zip(names, batch):
            for var_type, name in zip(predicate.var_types, names_per_lit):
                entities_list[batch_id].add(var_type(name))

    t.toc()

    return entities_list


def read_all_lits_and_probs_list(
    names_dict, batch_dict, probs_before_dict, probs_after_dict, predicates, **kwargs
):
    example_pred_name = list(predicates.keys())[0]
    batch_size = batch_dict[example_pred_name][-1].item() + 1

    t.tic("read all lits and probs")

    all_lits_list = [[] for _ in range(batch_size)]
    probs_before_list = [[] for _ in range(batch_size)]
    probs_after_list = [[] for _ in range(batch_size)]

    for pred_name in predicates.keys():
        predicate = predicates[pred_name]
        names, batch, probs_before, probs_after = (
            names_dict[pred_name],
            batch_dict[pred_name],
            probs_before_dict[pred_name].detach(),
            probs_after_dict[pred_name].detach(),
        )

        for names_per_lit, batch_id, prob_before, prob_after in zip(
            names, batch, probs_before, probs_after
        ):
            lit = predicate(
                *[var_type(name) for var_type, name in zip(predicate.var_types, names_per_lit)]
            )
            all_lits_list[batch_id].append(lit)
            probs_before_list[batch_id].append(prob_before.item())
            probs_after_list[batch_id].append(prob_after.item())
            # all_lits_list[batch_id].append(
            #     (lit, prob_before.item(), prob_after.item())
            # )

        t.toc()

    return all_lits_list, probs_before_list, probs_after_list


def convert_labels_to_literals(
    labels_dict, names_dict, batch_dict, predicates, all_lits_list, **kwargs
):
    example_pred_name = list(predicates.keys())[0]
    num_samples = labels_dict[example_pred_name].shape[0]

    t.tic("convert labels to literals")

    literals_list = [[set(all_lits) for _ in range(num_samples)] for all_lits in all_lits_list]
    # literals_list = [[set() for _ in range(num_samples)] for _ in range(batch_size)]

    for pred_name, predicate in predicates.items():
        labels_multi_samples, names, batch = (
            labels_dict[pred_name],
            names_dict[pred_name],
            batch_dict[pred_name],
        )

        for sample_idx, labels in enumerate(labels_multi_samples):
            for label, names_per_lit, batch_id in zip(labels, names, batch):
                lit = predicate(
                    *[var_type(name) for var_type, name in zip(predicate.var_types, names_per_lit)]
                )

                # only keep positive literals
                if label.item() == 0:
                    assert lit in literals_list[batch_id][sample_idx]
                    literals_list[batch_id][sample_idx].remove(lit)

    # Parallel(n_jobs=2, backend="threading")(
    #     delayed(convert_literals)(pred_name, sample_idx)
    #     for pred_name in predicates
    #     for sample_idx in range(num_samples)
    # )
    t.toc()

    # for lits_list in literals_list:
    #     for lits in lits_list:
    #         for lit in lits:
    #             if len(set(lit.variables)) != len(lit.variables):
    #                 pdb.set_trace()
    return literals_list


def convert_literals_to_labels(
    literals_list,
    names_dict,
    batch_dict,
    predicates,
    unconstr_lits_list=None,
    **kwargs,
):
    num_samples = len(literals_list[0])

    t.tic("convert literals to labels")
    # initialize
    labels_dict = {}
    idx_dict = {}
    mask_dict = {}
    for pred_name in predicates:
        names, batch = names_dict[pred_name], batch_dict[pred_name]
        labels_dict[pred_name] = torch.zeros(
            (num_samples, batch.shape[0]),
            device=batch.device,
            dtype=torch.float,
        )

        if unconstr_lits_list is not None:
            mask_dict[pred_name] = torch.ones(
                (num_samples, batch.shape[0]),
                device=batch.device,
                dtype=torch.bool,
            )

        idx_dict[pred_name] = {}
        for idx, (name_list, batch_id) in enumerate(zip(names, batch)):
            idx_dict[pred_name][batch_id.item(), tuple(name_list)] = idx

    # convert samples
    for i, literals_list_multi_samples in enumerate(literals_list):
        for j, lits in enumerate(literals_list_multi_samples):
            for lit in lits:
                pred_name = lit.predicate.name
                name_tuple = tuple([var.name for var in lit.variables])

                idx = idx_dict[pred_name][i, name_tuple]
                labels_dict[pred_name][j, idx] = 1

    # mask out unconstr lits
    if unconstr_lits_list is not None:
        for i, unconstr_literals_multi_samples in enumerate(unconstr_lits_list):
            for j, unconstr_lits in enumerate(unconstr_literals_multi_samples):
                for unconstr_lit in unconstr_lits:
                    pred_name = unconstr_lit.predicate.name
                    name_tuple = tuple([var.name for var in unconstr_lit.variables])

                    idx = idx_dict[pred_name][i, name_tuple]
                    mask_dict[pred_name][j, idx] = False

    t.toc()

    if unconstr_lits_list is not None:
        return labels_dict, mask_dict
    else:
        return labels_dict


# def convert_literals_to_labels(
#     literals_list,
#     names_dict,
#     batch_dict,
#     predicates,
#     unconstr_lits_list=None,
#     **kwargs,
# ):
#     num_samples = len(literals_list[0])

#     t.tic("convert literals to labels")
#     # initialize
#     labels_dict = {}
#     idx_dict = {}
#     mask_dict = {}
#     for pred_name in predicates:
#         names, batch = names_dict[pred_name], batch_dict[pred_name]
#         labels_dict[pred_name] = torch.zeros(
#             (num_samples, batch.shape[0]),
#             device=batch.device,
#             dtype=torch.float,
#         )

#         if unconstr_lits_list is not None:
#             mask_dict[pred_name] = torch.ones(
#                 (num_samples, batch.shape[0]),
#                 device=batch.device,
#                 dtype=torch.bool,
#             )

#         idx_dict[pred_name] = {}
#         for idx, (name_list, batch_id) in enumerate(zip(names, batch)):
#             idx_dict[pred_name][batch_id.item(), tuple(name_list)] = idx

#     # convert samples
#     for i, literals_list_multi_samples in enumerate(literals_list):
#         for j, lits in enumerate(literals_list_multi_samples):
#             for lit in lits:
#                 pred_name = lit.predicate.name
#                 name_tuple = tuple([var.name for var in lit.variables])

#                 idx = idx_dict[pred_name][i, name_tuple]
#                 labels_dict[pred_name][j, idx] = 1

#     # mask out unconstr lits
#     if unconstr_lits_list is not None:
#         for i, unconstr_literals_multi_samples in enumerate(unconstr_lits_list):
#             for j, unconstr_lits in enumerate(unconstr_literals_multi_samples):
#                 for unconstr_lit in unconstr_lits:
#                     pred_name = unconstr_lit.predicate.name
#                     name_tuple = tuple([var.name for var in unconstr_lit.variables])

#                     idx = idx_dict[pred_name][i, name_tuple]
#                     mask_dict[pred_name][j, idx] = False

#     t.toc()

#     if unconstr_lits_list is not None:
#         return labels_dict, mask_dict
#     else:
#         return labels_dict


def compute_valid_mask_dict(valid_mask, batch_dict, **kwargs):
    valid_mask_dict = {}
    for pred_name, batch in batch_dict.items():
        valid_mask_dict[pred_name] = valid_mask[:, batch]

    return valid_mask_dict


# def labels_from_literals(
#     literals_list, obs, predicates, predicate_to_idx, key_to_predicates, num_samples
# ):

#     for i, literals_list_multi_samples in enumerate(literals_list):
#         for j, lits in enumerate(literals_list_multi_samples):
#             if lits is None:
#                 continue

#             for lit in lits:
#                 concept_name = "_".join(lit.predicate.name.split("_")[:-1])
#                 name_tuple = tuple([var.name for var in lit.variables])
#                 region_id = int(lit.predicate.name.split("_")[-1])

#                 idx = concept_labels_dict[concept_name][1][i, name_tuple]
#                 concept_labels_dict[concept_name][2][j, idx] = region_id

#         assert lit.predicate.name in predicate_to_idx
#         key, idx = predicate_to_idx[lit.predicate.name]

#         # binary literals
#         if isinstance(key, tuple):
#             type_1, type_2 = key[0], key[2]
#             obj_idx_1 = data[type_1].names.index(lit.variables[0].name)
#             obj_idx_2 = data[type_2].names.index(lit.variables[1].name)
#             edge_idx = (
#                 (
#                     data[key].edge_index.transpose(0, 1)
#                     == torch.tensor([obj_idx_1, obj_idx_2])
#                 )
#                 .all(dim=1)
#                 .nonzero()
#             )

#             assert edge_idx.shape[0] == 1
#             data[key].gt_labels[edge_idx.item(), idx] = 1

#         # unary literals
#         else:
#             obj_idx = data[key].names.index(lit.variables[0].name)
#             data[key].gt_labels[obj_idx, idx] = 1

#     concept_labels_dict = {}
#     for concept in concepts:
#         last_labels, names, batch, _ = concept.read_labels_from_obs(obs)
#         labels = -1 * torch.ones(
#             (num_samples, *last_labels.shape[1:]),
#             device=batch.device,
#             dtype=torch.long,
#         )

#         idx_dict = {}
#         for idx, (name_list, batch_id) in enumerate(zip(names, batch)):
#             idx_dict[batch_id.item(), tuple(name_list)] = idx

#         int_labels = -1 * torch.ones(
#             (num_samples, *last_labels.shape[1:-1]),
#             device=batch.device,
#             dtype=torch.long,
#         )
#         concept_labels_dict[concept.concept_name] = (labels, idx_dict, int_labels)

#     for i, literals_list_multi_samples in enumerate(literals_list):
#         for j, lits in enumerate(literals_list_multi_samples):
#             if lits is None:
#                 continue

#             for lit in lits:
#                 concept_name = "_".join(lit.predicate.name.split("_")[:-1])
#                 name_tuple = tuple([var.name for var in lit.variables])
#                 region_id = int(lit.predicate.name.split("_")[-1])

#                 idx = concept_labels_dict[concept_name][1][i, name_tuple]
#                 concept_labels_dict[concept_name][2][j, idx] = region_id

#     # Replace back
#     for concept in concepts:
#         labels = concept_labels_dict[concept.concept_name][0]
#         int_labels = concept_labels_dict[concept.concept_name][2]
#         mask = int_labels != -1
#         labels[mask] = F.one_hot(int_labels[mask], num_classes=labels.shape[-1])
#         concept.add_labels_to_obs(labels, obs)

#     obs.num_samples = num_samples  # update number of samples
