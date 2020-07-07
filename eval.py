def get_entities(sequence_tag):
    #     entity = {
    #         'begin': 0,
    #         'end': 0,
    #         'type': ''
    #     }
    entities = []
    ne_type = ''
    is_ne = False
    flag = 0
    for i, tag in enumerate(sequence_tag):
        j = 0
        if 'B-' in tag:
            if is_ne:
                flag = 0
                for j in range(begin, i):
                    if ne_type != sequence_tag[j].split("-")[1]:
                        entities.append({'begin': begin, 'end': j, 'type': ne_type})
                        is_ne = False
                        ne_type = ''
                        flag = 1
                        break
                if not flag:
                    entities.append({'begin': begin, 'end': i, 'type': ne_type})
                    is_ne = False
                    ne_type = ''
            begin = i + 1
            ne_type = tag.split('-')[1]
            is_ne = True
        elif 'O' == tag and is_ne:
            flag = 0
            for j in range(begin, i):
                if ne_type != sequence_tag[j].split("-")[1]:
                    entities.append({'begin': begin, 'end': j, 'type': ne_type})
                    is_ne = False
                    ne_type = ''
                    flag = 1
                    break
            if not flag:
                entities.append({'begin': begin, 'end': i, 'type': ne_type})
                is_ne = False
                ne_type = ''
    if is_ne:
        entities.append({'begin': begin, 'end': i + 1, 'type': ne_type})
    return entities


def get_actual_predict_labels(filepath):
    predict = []
    actual = []
    with open(filepath, "r", encoding="utf8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            predict.append(line[-2])
            actual.append(line[-1])
    return predict, actual


def classify_entities_by_types(entities):
    per = []
    loc = []
    org = []
    misc = []

    for entity in entities:
        if entity["type"] == "PER":
            per.append(entity)
        if entity["type"] == "LOC":
            loc.append(entity)
        if entity["type"] == "LOC":
            org.append(entity)
        if entity["type"] == "MISC":
            misc.append(entity)
    return per, loc, org, misc


def compute_metrics_by_type(pre_entities, act_entites):
    correct = 0
    for pre_item in pre_entities:
        for act_item in act_entites:
            if pre_item == act_item:
                correct += 1
    precision = correct/len(pre_entities)
    recall = correct/len(act_entites)
    return precision, recall, 2*precision*recall/(precision+recall)



if __name__ == '__main__':
    labels = ["PER", "LOC", "ORG", "MISC"]
    filepath = "/home/swift/BERT-GAT-NER/output/9.txt"
    predict_labels, actual_labels = get_actual_predict_labels(filepath)

    predict_entities = get_entities(predict_labels)
    actual_entities = get_entities(actual_labels)

    pre_per, pre_loc, pre_org, pre_misc = classify_entities_by_types(predict_entities)
    act_per, act_loc, act_org, act_misc = classify_entities_by_types(actual_entities)

    per_scores = compute_metrics_by_type(pre_per, act_per)
    loc_scores = compute_metrics_by_type(pre_loc, act_loc)
    org_scores = compute_metrics_by_type(pre_org, act_org)
    misc_scores = compute_metrics_by_type(pre_misc, act_misc)

    print(per_scores)
    print(loc_scores)
    print(org_scores)
    print(misc_scores)

    P = (per_scores[0] + loc_scores[0] + org_scores[0] + misc_scores[0])/4
    R = (per_scores[1] + loc_scores[1] + org_scores[1] + misc_scores[1])/4
    print("Macro-F1: {}".format(2*P*R/(P+R)))

