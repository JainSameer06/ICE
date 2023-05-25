def bucketize_data(data, aspect, bucket_scores=[0.25, 0.5, 0.75, 1]):
    num_buckets = len(bucket_scores)
    buckets = [[] for i in range(num_buckets)]
    score_field = f'expert_{aspect}_mean'

    for example in data:
        for i, score in enumerate(bucket_scores):
            if example[score_field] <= score:
                buckets[i].append(example)
                break

    return buckets