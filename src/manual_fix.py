import argparse

from helper import get_profile_embeddings, get_profile_text
import os
import json
import ast

# add args


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default='movie', help='name of datasets')
parser.add_argument('--input', '-i', type=str, default='gema_movie_tuning.json', help='input file name for user profiles')
parser.add_argument('--output', '-o', type=str, default='full_gema_user_feat.npy', help='output file name for user profiles')
args, _ = parser.parse_known_args()

dataset = args.dataset
dir = f'./data/{dataset}/'
# =========================
# Load user data
# =========================
file_path = f'./data/{dataset}/{args.input}'

with open(file_path, 'r', encoding='utf-8') as jsonfile:
	jsonfile = json.load(jsonfile)


counter = 0
counter2 = 0
counter3 = 0
profile = []
import re, json
from tqdm import tqdm
for ii in tqdm(range(len(jsonfile))):
	try:
		raw = jsonfile[str(ii)]['summary']
		clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
		level = json.loads(clean)
		summarization = level["summarization"]
		reasoning = level["reasoning"]
	except Exception:
		try:
			level1 = ast.literal_eval(jsonfile[str(ii)]['summary'])
			level2 = ast.literal_eval(level1['summary'])
			summarization = level2['summarization']
			counter2 += 1
		except Exception:
			try:
				summarization = jsonfile[str(ii)]['summary']
				counter3 += 1
			except Exception:
				print(f"Error parsing JSON for key: {ii}")
				print(f"Value: {jsonfile[str(ii)]}")
	profile.append(summarization)
	counter += 1
print(f"Total successfully parsed profiles: {counter} , {counter2} with extra level, {counter3} with direct text")
print(f"Total profiles parsed with encoding errors: {counter2}")

# encode user profiles to embeddings and save as .npy
user_embeddings = get_profile_embeddings(profile, save = True, path = os.path.join(dir, f'{args.output}'))
print("User profile embeddings shape:", user_embeddings.shape)