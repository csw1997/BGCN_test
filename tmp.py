import json
from collections import Counter

with open("ace-05-splits/JMEE_dev.json", "r") as f:
    print("hello world!")
    raw_train_set = json.load(f)

'''
sent_argument_check = []
for sent_info in raw_train_set:
    entity_pos_list = []
    for entity_info in sent_info["golden-entity-mentions"]:
        entity_pos_list.append((entity_info["start"], entity_info["end"]))
    argument_pos_list = []
    for event_info in sent_info["golden-event-mentions"]:
        for argument_info in event_info["arguments"]:
            argument_pos_list.append((argument_info["start"], argument_info["end"]))
    
    check_not_pass = 0
    for argument_pos in argument_pos_list:
        if argument_pos not in entity_pos_list:
            check_not_pass += 1
    sent_argument_check.append([len(argument_pos_list), check_not_pass])

print(len(sent_argument_check))
print(len(list(filter(lambda x:x[1] , sent_argument_check))))
'''
with open("ace-05-splits/JMEE_dev.json", "r") as f:
    print("hello world!")
    raw_train_set = json.load(f)
for sent_info in raw_train_set:
    if len(sent_info["golden-entity-mentions"]) == 0:
        print("JMEE_dev", sent_info["sentence"])

with open("ace-05-splits/JMEE_train.json", "r") as f:
    print("hello world!")
    raw_train_set = json.load(f)
for sent_info in raw_train_set:
    if len(sent_info["golden-entity-mentions"]) == 0:
        print("JMEE_train", sent_info["sentence"])

with open("ace-05-splits/JMEE_test.json", "r") as f:
    print("hello world!")
    raw_train_set = json.load(f)
for sent_info in raw_train_set:
    if len(sent_info["golden-entity-mentions"]) == 0:
        print("JMEE_test", sent_info["sentence"])

