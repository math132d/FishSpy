def evaluate_iou(ground_truth, inferred, threshold) {
    tp = 0
    fp = 0
    fn = 0

    for inferred_slice in inferred:
        iou = 0

        for truth_slice in ground_truth:
            iou = inferred_slice.intersection_over_union(ground_truth)

            if(iou >= threshold):
                tp += 1
                break;

        if(iou == 0):
            fp += 1

    fn = len(ground_truth) - tp;

    precision = tp / (tp + fp)
    recall = tp / (tp + tn)

    return (tp, fp, fn, precision, recall)
}