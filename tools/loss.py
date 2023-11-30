import torch


def shift_window_triple_loss(args, query_feature, database_feature, loss_fn):
    """
    Calculate the shift window triple loss for a given query feature against a database feature.

    This function computes the similarity between a query feature and different windows of a database feature.
    It uses a shift window approach where the window moves across the database feature in steps defined by
    the reduce_factor in args. The loss is calculated using the provided loss function (loss_fn) which takes
    the query feature, a positive match from the database, and a negative match from the database.

    Parameters:
    - args (Namespace): A namespace or similar object containing configuration parameters, including
      'reduce_factor' which determines the step size for window shifting.
    - query_feature (torch.Tensor): A tensor representing the query feature.
    - database_feature (torch.Tensor): A tensor representing the database feature against which the query
      is compared.
    - loss_fn (function): A loss function that takes three arguments (query_feature, positive_feature,
      negative_feature) and returns a scalar loss value.

    Returns:
    - loss_sum (float): The accumulated loss over all the shifted windows.
    - min_index_in_row (torch.Tensor): The indices of the minimum values in the maintain_table, which
      represent the most similar window of database_feature to the query_feature.

    The function operates by sliding a window across the database feature and computing a similarity
    measure between the query feature and each window using L2 norm. It identifies the most similar
    window (positive match) and uses other windows as negative matches to compute the loss. The process
    is repeated for each window, and the losses are summed up to get the total loss.

    The function assumes that the query_feature and database_feature have compatible shapes and that the
    loss_fn is properly defined to handle the inputs.

    Example:
    >>> loss, indices = shift_window_triple_loss(args, query_feature, database_feature, loss_fn)
    """
    window_size = query_feature.shape[0]
    step = int(window_size/args.reduce_factor)
    left, right = 0, window_size
    loss_sum = 0.

    split_similarity_list = []
    width_bound = database_feature.shape[1]
    while left < width_bound:
        if right <= width_bound:
            split_similarity = torch.norm(query_feature - database_feature[:, left:right], p=2, dim=1)
        else:
            cycle = torch.concat((database_feature[:,left:],database_feature[:,:right-width_bound]),dim=1)
            split_similarity = torch.norm(query_feature-cycle,p=2,dim=1)
        split_similarity_list.append(split_similarity)

        left += step
        right += step

    # Similarity matrix for each split window and get the most similar slice_index using L2 Norm
    maintain_table = torch.transpose(torch.stack(split_similarity_list), 0, 1)  # shape:11, split_nums
    maintain_table_aggregation, min_index_in_row = torch.min(maintain_table, dim=1)

    # 0 -> positive and slice long vector according to slice_index above
    if min_index_in_row[0] * step + window_size <= width_bound:
        filtered_positive = database_feature[0,
                            min_index_in_row[0] * step:min_index_in_row[0] * step + window_size]  # shape:feature_dim
    else:
        filtered_positive = torch.concat((database_feature[0,min_index_in_row[0] * step:],
                                          database_feature[0,:min_index_in_row[0] * step + window_size - width_bound]),
                                         dim=-1)


    for i in range(1, 11):
        # 1 -> 10 refer to negs
        if min_index_in_row[i] * step + window_size <=width_bound:
            filtered_negative = database_feature[i,
                                min_index_in_row[i] * step:min_index_in_row[i] * step + window_size]  # shape:feature_dim
        else:
            filtered_negative = torch.concat((database_feature[i,min_index_in_row[i] * step:],
                                              database_feature[i,:min_index_in_row[i] * step + window_size - width_bound]),
                                             dim=-1)

        loss_sum += loss_fn(query_feature, filtered_positive, filtered_negative)

    return loss_sum, min_index_in_row

