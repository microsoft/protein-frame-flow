import numpy as np
def fixed_idcs_mask_from_index_mapping_and_redesign_idcs(
        source_idx_to_full_idx, contig, redesign_idcs, L):
    """
    Given a mapping from source indices to full indices, and a string of redesign indices,
    return a boolean mask of the full indices that are fixed.

    Args:
        source_idx_to_full_idx (dict): mapping from source indices to full indices
        contig (str): string of contig, e.g. "5-20,A16-35,10-25,A52-71,5-20"
        redesign_idcs (str): string of redesign indices, e.g. "A16-19,A21,A23,A25,A27-30"
    """
    redesign_mask = np.ones(L, dtype=int) # assume all are redesignable by default

    # pull out ranges in the contig in chain A, based on if they contain a Letter
    ranges = contig.split(",")
    ranges = [r for r in ranges if r[0].isupper()]

    # Loop through ranges and set redesign_mask to 0 for indices in the range
    for range_ in ranges:
        chain, source_range = range_[0], range_[1:]
        if source_range.count("-"):
            source_st, source_end = source_range.split("-")
        else:
            source_st = source_range
            source_end = source_range
        source_st, source_end = int(source_st), int(source_end)
        source_idcs = np.arange(source_st, source_end)
        for source_idx in source_idcs:
            full_idx = source_idx_to_full_idx[chain + str(source_idx)]
            redesign_mask[full_idx] = 0.

    # And then overwrite with the indices in the redesign_idcs back to 1.
    for redesign_idx in redesign_idcs.split(","):
        chain, source_range = redesign_idx[0], redesign_idx[1:]
        if source_range.count("-"):
            source_st, source_end = source_range.split("-")
        else:
            source_st = source_range
            source_end = source_range
        source_st, source_end = int(source_st), int(source_end)
        source_idcs = np.arange(source_st, source_end)
        for source_idx in source_idcs:
            if not chain + str(source_idx) in source_idx_to_full_idx:
                print(chain + str(source_idx), "not in source_idx_to_full_idx")
                continue
            full_idx = source_idx_to_full_idx[chain + str(source_idx)]
            redesign_mask[full_idx] = 1.
    fixed_mask = 1. - redesign_mask
    return fixed_mask

def motif_locs_and_contig_to_fixed_idcs_mask(motif_locs, contig, redesign_idcs, L):
    """
    Given the motif locations and the contig, return a boolean mask of the full indices that are fixed.


    Args:
        motif_locs (list): list of tuples of motif locations, e.g. [(5, 20), (62, 77)]
        contig (str): string of contig, e.g. "5-20,A16-35,10-25,A52-71,5-20"
        redesign_idcs (str): string of redesign indices, e.g. "A16-19,A21,A23,A25,A27-30"
        L (int): length of the full protein
    """
    ranges = contig.split(",")
    ranges = [r for r in ranges if r[0].isupper()]
    source_idx_to_full_idx = {}
    for i, (st, _) in enumerate(motif_locs):
        chain, source_range = ranges[i][0], ranges[i][1:]
        if "-" in source_range:
            segment_st, segment_end = source_range.split("-")
        else:
            segment_st = segment_end = source_range
        segment_st, segment_end = int(segment_st), int(segment_end)
        for source_idx in range(segment_st, segment_end):
            full_idx = st + source_idx - segment_st
            key = chain + str(source_idx)
            source_idx_to_full_idx[key] = full_idx

    fixed_idcs_mask = fixed_idcs_mask_from_index_mapping_and_redesign_idcs(
        source_idx_to_full_idx, contig, redesign_idcs, L)
    return fixed_idcs_mask

def seq_indices_to_fix(motif_locs, contig, redesign_idcs, L=None):
    if L == None:
        L = 2*max([end for _, end in motif_locs])
    mask = motif_locs_and_contig_to_fixed_idcs_mask(motif_locs, contig, redesign_idcs, L)
    idcs = [i for i, m in enumerate(mask) if m == 1]
    return idcs