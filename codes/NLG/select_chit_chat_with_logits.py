import json
import numpy as np
import sys

def read_data(file):
	with open(file, 'r', encoding="utf-8") as reader:
		data = json.load(reader)
	return data

test_seen = read_data(sys.argv[1])

select_ratio = 0.55
select_num = int(select_ratio * len(test_seen))

test_seen_temp = []
for i in test_seen:
	test_seen_temp.append((i['idx'], i['logits']))

test_seen_temp.sort(key = lambda x : x[1], reverse = True)
select_true = [test_seen_temp[i][0] for i in range(select_num)]

final_output = {}
for i,j in enumerate(test_seen):
	if j['id'] in final_output:
		if j['turn_id'] in final_output[j['id']]:
			if j['idx'] in select_true and "welcome" not in j['chit-chat'].lower():
				if j['location'] == 'begin':
					final_output[j['id']][j['turn_id']]['start'] = j['chit-chat']
				else:
					final_output[j['id']][j['turn_id']]['end'] = j['chit-chat']
		else:
			final_output[j['id']][j['turn_id']] = {"start": '', "end": '', "mod": ''}
			if j['idx'] in select_true and "welcome" not in j['chit-chat'].lower():
				if j['location'] == 'begin':
					final_output[j['id']][j['turn_id']]['start'] = j['chit-chat']
				else:
					final_output[j['id']][j['turn_id']]['end'] = j['chit-chat']
	else:
		final_output[j['id']] = {}
		final_output[j['id']][j['turn_id']] = {"start": '', "end": '', "mod": ''}
		if j['idx'] in select_true and "welcome" not in j['chit-chat'].lower():
			if j['location'] == 'begin':
				final_output[j['id']][j['turn_id']]['start'] = j['chit-chat']
			else:
				final_output[j['id']][j['turn_id']]['end'] = j['chit-chat']

with open(sys.argv[2], 'w') as f:
	json.dump(final_output, f, indent=2)


